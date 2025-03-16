import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from flask import Flask, jsonify, send_from_directory, abort, request
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

# Global flag for controlling streaming
streaming_active = True

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
        # Try different camera indices for macOS
        for camera_index in [0, 1]:
            try:
                camera_src = str(camera_index)
                smart_cctv = SmartCCTV(camera_source=camera_src)
                video_stream = smart_cctv.vs  # Get the VideoStream from SmartCCTV
                
                # Test if we can get a frame
                test_frame = video_stream.read_latest()
                if test_frame is not None:
                    logger.info(f"Successfully initialized SmartCCTV with camera index {camera_index}")
                    return True
            except Exception as e:
                logger.warning(f"Failed to initialize SmartCCTV with camera index {camera_index}: {str(e)}")
                if smart_cctv:
                    smart_cctv.cleanup()
                    
        logger.error("No working camera found")
        return False
    except Exception as e:
        logger.error(f"SmartCCTV initialization failed: {str(e)}")
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
                    socketio.emit('incident_detected', {
                        'timestamp': int(current_time * 1000),
                        'message': 'Incident detected!'
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