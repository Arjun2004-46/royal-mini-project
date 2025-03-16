import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from flask import Flask, jsonify, send_from_directory, abort, request
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
import numpy as np
import base64
from smart_cctv import VideoStream, IncidentRecorder
import threading
import time
import os
import json
from datetime import datetime
import logging
import uuid

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

# Initialize video stream with lower resolution
video_stream = VideoStream(src=0)
video_stream.start()
incident_recorder = IncidentRecorder("incidents")

# Frame processing settings
FRAME_RATE = 15  # Reduced from 24 to 15 FPS for better performance
JPEG_QUALITY = 40  # Reduced from 60 to 40 for better performance
MAX_WIDTH = 480  # Reduced from 640 to 480 for better performance
FRAME_SKIP_THRESHOLD = 0.1  # Skip frames if processing takes longer than this

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
        # Convert to grayscale for better performance (optional)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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

def stream_video():
    """Stream video frames through WebSocket with rate limiting and frame skipping"""
    frame_interval = 1.0 / FRAME_RATE
    last_frame_time = time.time()
    frames_processed = 0
    frames_skipped = 0
    last_stats_time = time.time()
    frame_number = 0
    
    while streaming_active:
        try:
            current_time = time.time()
            
            # Calculate actual processing time
            processing_time = current_time - last_frame_time
            
            # Skip frame if processing is too slow
            if processing_time < FRAME_SKIP_THRESHOLD:
                frame = video_stream.read_latest()
                if frame is not None:
                    # Process and emit frame
                    encoded_frame = encode_frame(frame)
                    if encoded_frame:
                        try:
                            frame_number += 1
                            socketio.emit('video_frame', {
                                'frame': encoded_frame,
                                'timestamp': int(current_time * 1000),  # milliseconds
                                'frame_number': frame_number
                            })
                            frames_processed += 1
                            last_frame_time = current_time
                        except Exception as e:
                            logger.error(f"Error emitting frame: {str(e)}")
                else:
                    logger.warning("No frame available from camera")
                    time.sleep(0.1)  # Short sleep when no frame is available
            else:
                frames_skipped += 1
                logger.debug(f"Skipped frame, processing time: {processing_time:.3f}s")
                
            # Log performance stats every 5 seconds
            if current_time - last_stats_time >= 5:
                actual_fps = frames_processed / (current_time - last_stats_time)
                skip_ratio = frames_skipped / (frames_processed + frames_skipped) if frames_processed + frames_skipped > 0 else 0
                logger.info(f"Streaming stats - FPS: {actual_fps:.1f}, Skipped: {frames_skipped}, Skip ratio: {skip_ratio:.2%}")
                frames_processed = 0
                frames_skipped = 0
                last_stats_time = current_time
                
            # Dynamic sleep to maintain frame rate
            elapsed = time.time() - current_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                    
        except Exception as e:
            logger.error(f"Error in stream_video: {str(e)}")
            time.sleep(0.1)  # Short sleep on error

@app.route('/api/incidents')
def get_incidents():
    """Get list of incidents"""
    try:
        return jsonify(load_incidents())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/incidents/<incident_uuid>')
def get_incident_detail(incident_uuid):
    """Get detailed information about a specific incident"""
    try:
        incidents = load_incidents()
        incident = next((inc for inc in incidents if inc.get('uuid') == incident_uuid), None)
        
        if not incident:
            return jsonify({'error': 'Incident not found'}), 404
            
        # Add the image data to the incident details
        image_path = os.path.join('incidents', incident['image'])
        if os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
                incident['image_data'] = f"data:image/jpeg;base64,{image_data}"
        
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
        encoded_frame = encode_frame(frame)
        if encoded_frame:
            socketio.emit('video_frame', {
                'frame': encoded_frame,
                'timestamp': int(time.time() * 1000),
                'frame_number': 0
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
        # Start streaming thread
        streaming_thread = start_streaming()
        # Run the web server
        socketio.run(app, host='0.0.0.0', port=PORT, debug=True)
    except KeyboardInterrupt:
        print("\nShutting down the server...")
        streaming_active = False  # Signal the streaming thread to stop
        video_stream.stop()
        time.sleep(1)  # Give the streaming thread time to clean up 