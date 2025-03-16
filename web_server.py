import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from flask import Flask, jsonify, send_from_directory, abort, request, Response
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
import numpy as np
import base64
from smart_cctv import SmartCCTV, VideoStream, IncidentRecorder
import threading
import time
import os
import json
from datetime import datetime
import logging
import uuid
import sys
import queue
import requests

# Load configuration
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("config.json not found. Using default configuration.")
        return {
            'DETECTION': {
                'PROCESS_EVERY_N_FRAMES': 2,
                'MIN_INCIDENT_INTERVAL': 3,
                'MAX_WORKERS': 4
            },
            'CAMERA': {
                'WIDTH': 1280,
                'HEIGHT': 720,
                'FPS': 30,
                'BUFFER_SIZE': 1
            }
        }
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config.json: {str(e)}")
        return None

CONFIG = load_config()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode='eventlet',
                   ping_timeout=10,
                   ping_interval=5,
                   logger=True,           # Enable socket.io logging
                   engineio_logger=True)  # Enable engine.io logging

# Initialize components
smart_cctv = None
video_stream = None
incident_recorder = IncidentRecorder("incidents")

# Frame processing settings
FRAME_RATE = 15  # Reduced from 24 to 15 FPS for better performance
JPEG_QUALITY = 30  # Reduced from 40 to 30 for better performance
MAX_WIDTH = 640  # Increased for better quality
FRAME_SKIP_THRESHOLD = 0.066  # ~15 FPS (1/15)

# Server configuration
PORT = 5001  # Changed from default 5000 to avoid conflicts
API_PORT = 5002  # Port for the API server

# Global flag for controlling streaming
streaming_active = True

# Add notification queue for SSE
notification_queue = queue.Queue()

def send_sse_notification(notification_type, data):
    """Helper function to send SSE notifications"""
    notification = {
        'type': notification_type,
        'data': data,
        'timestamp': int(time.time() * 1000)
    }
    notification_queue.put(notification)

# Function to handle fire alerts from SmartCCTV
def handle_fire_alert(alert_type, metadata):
    """Handle fire alerts from the SmartCCTV system and notify clients"""
    logger.info(f"Fire alert callback triggered with type: {alert_type} and metadata: {metadata}")
    
    if alert_type == "fire" and metadata:
        try:
            # Prepare notification data
            notification_data = {
                'message': 'FIRE DETECTED!',
                'confidence': metadata.get('confidence', 0.0),
                'threshold': metadata.get('threshold', 0.4),
                'severity': metadata.get('severity', 'medium'),
                'timestamp': int(time.time() * 1000)
            }
            
            logger.info(f"Preparing to send notification: {notification_data}")
            
            # Add notification to file
            result = add_notification('fire_alert', notification_data)
            if result:
                logger.info(f"Fire alert notification successfully saved with ID: {result.get('id', 'unknown')}")
            else:
                logger.error("Failed to save fire alert notification - no result returned")
            
        except Exception as e:
            logger.error(f"Error saving fire alert: {str(e)}", exc_info=True)

# Function to handle fall alerts from SmartCCTV
def handle_fall_alert(alert_type, metadata):
    """Handle fall alerts from the SmartCCTV system and notify clients"""
    if alert_type == "fall" and metadata:
        try:
            # Prepare notification data
            notification_data = {
                'message': 'FALL DETECTED!',
                'confidence': metadata.get('confidence', 0.0),
                'severity': metadata.get('severity', 'medium'),
                'timestamp': int(time.time() * 1000)
            }
            
            # Add notification to file
            add_notification('fall_alert', notification_data)
            logger.info(f"Fall alert notification saved: {metadata.get('confidence', 0.0):.2f} confidence")
            
        except Exception as e:
            logger.error(f"Error saving fall alert: {str(e)}")

def add_notification(notification_type, data):
    """Add a notification to the notifications file"""
    try:
        # Add retry mechanism for API server connection
        max_retries = 3
        retry_delay = 1  # seconds
        
        logger.info(f"Attempting to send notification to API server at http://localhost:{API_PORT}/api/notifications")
        logger.info(f"Notification data: type={notification_type}, data={data}")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} to send notification")
                response = requests.post(
                    f'http://localhost:{API_PORT}/api/notifications',
                    json={'type': notification_type, 'data': data},
                    timeout=5  # 5 seconds timeout
                )
                logger.info(f"API response status code: {response.status_code}")
                response.raise_for_status()
                response_data = response.json()
                logger.info(f"Successfully added notification to API server: {response_data}")
                return response_data
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise Exception(f"Failed to connect to API server after {max_retries} attempts: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error adding notification: {str(e)}", exc_info=True)
        # Fallback: Send through SSE if API server is not available
        logger.info("Falling back to SSE notification")
        send_sse_notification(notification_type, data)
        return None

# Function to load incidents with UUID if not present
def load_incidents():
    try:
        incidents_file = os.path.join('incidents', 'incident_log.json')
        if os.path.exists(incidents_file):
            with open(incidents_file, 'r') as f:
                incidents = json.load(f)
                # Add UUID to existing incidents if they don't have one
                modified = False
                for incident in incidents:
                    if 'uuid' not in incident:
                        incident['uuid'] = str(uuid.uuid4())
                        modified = True
                if modified:
                    with open(incidents_file, 'w') as f:
                        json.dump(incidents, f, indent=2)
                return incidents
        return []
    except Exception as e:
        logger.error(f"Error loading incidents: {str(e)}")
        return []

# Load incidents at startup
incidents = load_incidents()

def resize_frame(frame):
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    if width > MAX_WIDTH:
        ratio = MAX_WIDTH / width
        new_height = int(height * ratio)
        frame = cv2.resize(frame, (MAX_WIDTH, new_height), interpolation=cv2.INTER_NEAREST)
    return frame

def encode_frame(frame):
    """Convert OpenCV frame to base64 string with optimized settings"""
    try:
        # Resize frame if needed
        frame = resize_frame(frame)
        
        # Encode with optimized settings
        encode_param = [
            cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,
        ]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding frame: {str(e)}")
        return None

def init_smart_cctv():
    """Initialize the SmartCCTV system"""
    global smart_cctv, video_stream
    try:
        logger.info("Starting SmartCCTV initialization...")
        # Try different camera indices for macOS
        for camera_index in [0, 1]:
            try:
                camera_src = str(camera_index)
                logger.info(f"Attempting to initialize camera with index {camera_index}")
                smart_cctv = SmartCCTV(camera_source=camera_src)
                video_stream = smart_cctv.vs  # Get the VideoStream from SmartCCTV
                
                # Test if we can get a frame
                test_frame = video_stream.read_latest()
                if test_frame is not None:
                    logger.info("Successfully got test frame from camera")
                    # Register alert callbacks
                    logger.info("Registering alert callbacks...")
                    smart_cctv.alert_system.register_alert_callback("fire", handle_fire_alert)
                    smart_cctv.alert_system.register_alert_callback("fall", handle_fall_alert)
                    logger.info(f"Successfully initialized SmartCCTV with camera index {camera_index}")
                    logger.info("Alert callbacks registered for: fire, fall")
                    return True
            except Exception as e:
                logger.warning(f"Failed to initialize SmartCCTV with camera index {camera_index}: {str(e)}", exc_info=True)
                if smart_cctv:
                    smart_cctv.cleanup()
                    
        logger.error("No working camera found")
        return False
    except Exception as e:
        logger.error(f"SmartCCTV initialization failed: {str(e)}", exc_info=True)
        return False

def stream_video():
    """Stream video frames through WebSocket with incident detection"""
    frame_interval = 1.0 / FRAME_RATE
    last_frame_time = time.time()
    frames_processed = 0
    frames_skipped = 0
    last_stats_time = time.time()
    frame_number = 0
    last_emit_time = 0
    min_emit_interval = 1.0 / FRAME_RATE
    
    while streaming_active:
        try:
            current_time = time.time()
            
            # Rate limiting
            if current_time - last_emit_time < min_emit_interval:
                time.sleep(0.001)
                continue
            
            # Get frame from video stream
            frame = video_stream.read_latest()
            if frame is not None:
                # Process frame with SmartCCTV
                processed_frame, incident_detected = smart_cctv.process_frame(frame.copy())
                
                # If an incident was detected, notify clients
                if incident_detected:
                    # Check if it's a fire incident by examining the confidence value
                    is_fire = False
                    fire_confidence = 0.0
                    
                    if hasattr(smart_cctv, 'fire_params') and smart_cctv.fire_params['confidence'] >= smart_cctv.fire_params['confidence_threshold']:
                        is_fire = True
                        fire_confidence = smart_cctv.fire_params['confidence']
                    
                    # Send detailed alert information
                    socketio.emit('incident_detected', {
                        'timestamp': int(current_time * 1000),
                        'message': 'Fire detected!' if is_fire else 'Incident detected!',
                        'type': 'fire' if is_fire else 'incident',
                        'confidence': fire_confidence if is_fire else 0.0,
                        'severity': 'high' if is_fire and fire_confidence > 0.7 else 'medium'
                    })
                
                # Encode and emit the processed frame
                encoded_frame = encode_frame(processed_frame)
                if encoded_frame:
                    try:
                        frame_number += 1
                        socketio.emit('video_frame', {
                            'frame': encoded_frame,
                            'timestamp': int(current_time * 1000),
                            'frame_number': frame_number,
                            'quality': JPEG_QUALITY,
                            'incident_detected': incident_detected
                        })
                        
                        frames_processed += 1
                        last_emit_time = current_time
                        last_frame_time = current_time
                        
                        # Calculate processing time
                        processing_time = time.time() - current_time
                        if processing_time > FRAME_SKIP_THRESHOLD:
                            frames_skipped += 1
                            logger.debug(f"Frame took too long to process: {processing_time:.3f}s")
                            
                    except Exception as e:
                        logger.error(f"Error emitting frame: {str(e)}")
            else:
                logger.warning("No frame available from camera")
                if not init_smart_cctv():
                    time.sleep(1.0)
                continue
                
            # Log performance stats every 5 seconds
            if current_time - last_stats_time >= 5:
                elapsed = current_time - last_stats_time
                actual_fps = frames_processed / elapsed if elapsed > 0 else 0
                skip_ratio = frames_skipped / (frames_processed + frames_skipped) if frames_processed + frames_skipped > 0 else 0
                logger.info(f"Streaming stats - FPS: {actual_fps:.1f}, Processed: {frames_processed}, Skipped: {frames_skipped}, Skip ratio: {skip_ratio:.2%}")
                frames_processed = 0
                frames_skipped = 0
                last_stats_time = current_time
                
        except Exception as e:
            logger.error(f"Error in stream_video: {str(e)}")
            time.sleep(0.1)

@app.route('/api/incidents')
def get_incidents():
    """Get list of incidents"""
    try:
        # Load incidents from file
        incidents = load_incidents()
        
        # Return a simplified list for better performance
        simplified_incidents = []
        for incident in incidents:
            simplified_incidents.append({
                'uuid': incident.get('uuid', ''),
                'type': incident.get('type', ''),
                'timestamp': incident.get('timestamp', ''),
                'confidence': incident.get('confidence', 0)
            })
        
        return jsonify(simplified_incidents)
    except Exception as e:
        logger.error(f"Error getting incidents list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/incidents/<incident_uuid>')
def get_incident_detail(incident_uuid):
    """Get detailed information about a specific incident"""
    try:
        logger.info(f"Fetching incident detail for UUID: {incident_uuid}")
        incidents = load_incidents()
        logger.info(f"Loaded {len(incidents)} incidents from file")
        
        # Debug: Log the first few UUIDs to compare
        if len(incidents) > 0:
            logger.info(f"First few UUIDs in the list: {[inc.get('uuid', 'NO_UUID') for inc in incidents[:3]]}")
        
        incident = next((inc for inc in incidents if inc.get('uuid') == incident_uuid), None)
        
        if not incident:
            logger.error(f"Incident not found with UUID: {incident_uuid}")
            return jsonify({'error': 'Incident not found'}), 404
            
        logger.info(f"Found incident: {incident.get('type')} at {incident.get('timestamp')}")
        
        # Add the image data to the incident details
        image_path = os.path.join('incidents', incident['image'])
        if os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
                incident['image_data'] = f"data:image/jpeg;base64,{image_data}"
                logger.info(f"Added image data from {image_path}")
        else:
            logger.warning(f"Image file not found: {image_path}")
        
        return jsonify(incident)
    except Exception as e:
        logger.error(f"Error getting incident detail: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/incidents/<incident_uuid>/image')
def get_incident_image_by_uuid(incident_uuid):
    """Get incident image by UUID"""
    try:
        incidents = load_incidents()
        incident = next((inc for inc in incidents if inc.get('uuid') == incident_uuid), None)
        
        if not incident:
            return jsonify({'error': 'Incident not found'}), 404
            
        image_path = os.path.join('incidents', incident['image'])
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
            
        # Get the requested format from query parameter (default to JPEG)
        image_format = request.args.get('format', 'jpeg').lower()
        
        if image_format == 'base64':
            # Return base64 encoded image
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
                return jsonify({
                    'image_data': f"data:image/jpeg;base64,{image_data}",
                    'filename': incident['image']
                })
        else:
            # Return the actual image file
            return send_from_directory('incidents', incident['image'], 
                                    mimetype='image/jpeg',
                                    as_attachment=request.args.get('download', 'false').lower() == 'true')
            
    except Exception as e:
        logger.error(f"Error getting incident image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/incidents/files/<path:filename>')
def get_incident_image(filename):
    """Serve incident images by filename (legacy endpoint)"""
    try:
        if not os.path.exists(os.path.join('incidents', filename)):
            return jsonify({'error': 'Image file not found'}), 404
            
        return send_from_directory('incidents', filename, 
                                 mimetype='image/jpeg',
                                 as_attachment=request.args.get('download', 'false').lower() == 'true')
    except Exception as e:
        logger.error(f"Error serving incident image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    # Send initial frame to client immediately
    frame = video_stream.read_latest()
    if frame is not None:
        # Process the frame with SmartCCTV
        processed_frame, incident_detected = smart_cctv.process_frame(frame.copy())
        encoded_frame = encode_frame(processed_frame)
        if encoded_frame:
            socketio.emit('video_frame', {
                'frame': encoded_frame,
                'timestamp': int(time.time() * 1000),
                'frame_number': 0,
                'quality': JPEG_QUALITY,
                'incident_detected': incident_detected
            })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on_error_default
def default_error_handler(e):
    """Handle Socket.IO errors"""
    logger.error(f'SocketIO error: {str(e)}')

def start_streaming():
    """Start the video streaming thread"""
    global streaming_active
    streaming_active = True
    streaming_thread = threading.Thread(target=stream_video)
    streaming_thread.daemon = True
    streaming_thread.start()
    return streaming_thread

@app.route('/api/notifications/stream')
def notification_stream():
    """SSE endpoint for real-time notifications"""
    def generate():
        while True:
            try:
                # Get notification from queue with timeout
                notification = notification_queue.get(timeout=30)
                # Format as SSE message
                data = f"data: {json.dumps(notification)}\n\n"
                yield data
            except queue.Empty:
                # Send keepalive comment to maintain connection
                yield ": keepalive\n\n"
            except Exception as e:
                logger.error(f"Error in notification stream: {str(e)}")
                break
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/')
def index():
    """Serve the notifications page"""
    return send_from_directory('.', 'notifications.html')

if __name__ == '__main__':
    try:
        # Initialize SmartCCTV
        if not init_smart_cctv():
            logger.error("Failed to initialize SmartCCTV. Exiting.")
            sys.exit(1)
            
        # Start streaming thread
        streaming_thread = start_streaming()
        # Run the web server
        socketio.run(app, host='0.0.0.0', port=PORT, debug=True)
    except KeyboardInterrupt:
        print("\nShutting down the server...")
        streaming_active = False  # Signal the streaming thread to stop
        if smart_cctv:
            smart_cctv.cleanup()
        time.sleep(1)  # Give the streaming thread time to clean up 