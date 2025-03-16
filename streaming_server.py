import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
import base64
from smart_cctv import SmartCCTV, VideoStream
import threading
import time
import logging
import sys

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
                   logger=True,
                   engineio_logger=True)

# Initialize components
smart_cctv = None
video_stream = None

# Frame processing settings
FRAME_RATE = 15
JPEG_QUALITY = 30
MAX_WIDTH = 640
FRAME_SKIP_THRESHOLD = 0.066

# Server configuration
STREAMING_PORT = 5001

# Global flag for controlling streaming
streaming_active = True

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
        frame = resize_frame(frame)
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
        for camera_index in [0, 1]:
            try:
                camera_src = str(camera_index)
                smart_cctv = SmartCCTV(camera_source=camera_src)
                video_stream = smart_cctv.vs
                
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
            
            if current_time - last_emit_time < min_emit_interval:
                time.sleep(0.001)
                continue
            
            frame = video_stream.read_latest()
            if frame is not None:
                processed_frame, incident_detected = smart_cctv.process_frame(frame.copy())
                
                if incident_detected:
                    socketio.emit('incident_detected', {
                        'timestamp': int(current_time * 1000),
                        'message': 'Incident detected!'
                    })
                
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

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    frame = video_stream.read_latest()
    if frame is not None:
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
        if not init_smart_cctv():
            logger.error("Failed to initialize SmartCCTV. Exiting.")
            sys.exit(1)
            
        streaming_thread = start_streaming()
        socketio.run(app, host='0.0.0.0', port=STREAMING_PORT, debug=True)
    except KeyboardInterrupt:
        print("\nShutting down the streaming server...")
        streaming_active = False
        if smart_cctv:
            smart_cctv.cleanup()
        time.sleep(1) 