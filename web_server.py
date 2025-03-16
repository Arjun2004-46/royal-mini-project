import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from flask import Flask, jsonify, send_from_directory
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode='eventlet',
                   ping_timeout=10,
                   ping_interval=5)

# Initialize video stream with lower resolution
video_stream = VideoStream(src=0)
video_stream.start()
incident_recorder = IncidentRecorder("incidents")

# Frame processing settings
FRAME_RATE = 24  # Reduced from 30 to 24 FPS
JPEG_QUALITY = 60  # Reduced from 80 to 60 for better performance
MAX_WIDTH = 640  # Maximum frame width

def resize_frame(frame):
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    if width > MAX_WIDTH:
        ratio = MAX_WIDTH / width
        new_height = int(height * ratio)
        frame = cv2.resize(frame, (MAX_WIDTH, new_height), interpolation=cv2.INTER_AREA)
    return frame

def encode_frame(frame):
    """Convert OpenCV frame to base64 string with optimized settings"""
    try:
        # Resize frame if needed
        frame = resize_frame(frame)
        
        # Encode with optimized settings
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding frame: {str(e)}")
        return None

def stream_video():
    """Stream video frames through WebSocket with rate limiting"""
    frame_interval = 1.0 / FRAME_RATE
    last_frame_time = time.time()
    
    while True:
        try:
            current_time = time.time()
            # Check if it's time to process the next frame
            if current_time - last_frame_time >= frame_interval:
                frame = video_stream.read_latest()
                if frame is not None:
                    encoded_frame = encode_frame(frame)
                    if encoded_frame:
                        socketio.emit('video_frame', encoded_frame)
                    last_frame_time = current_time
                
                # Dynamic sleep to maintain frame rate
                processing_time = time.time() - current_time
                sleep_time = max(0, frame_interval - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            logger.error(f"Error in stream_video: {str(e)}")
            time.sleep(0.1)  # Shorter recovery time

@app.route('/api/incidents')
def get_incidents():
    """Get list of incidents"""
    try:
        with open(os.path.join('incidents', 'incident_log.json'), 'r') as f:
            incidents = json.load(f)
        return jsonify(incidents)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/incidents/<path:filename>')
def get_incident_image(filename):
    """Serve incident images"""
    return send_from_directory('incidents', filename)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def start_streaming():
    """Start the video streaming thread"""
    streaming_thread = threading.Thread(target=stream_video)
    streaming_thread.daemon = True
    streaming_thread.start()

if __name__ == '__main__':
    # Start streaming thread
    start_streaming()
    # Run the web server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 