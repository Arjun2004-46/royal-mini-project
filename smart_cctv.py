import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from datetime import datetime
import os
import argparse
from threading import Thread, Lock, Event
import queue
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
import json
import pygame  # For audio alerts

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('cctv.log'),
        logging.StreamHandler()
    ]
)

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        
        # Enhanced camera properties
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increased resolution
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
        self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
        
        # Advanced threading
        self.q = queue.Queue(maxsize=4)
        self.lock = Lock()
        self.stopped = Event()
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        self.last_frame = None
        self.frame_ready = Event()

    def start(self):
        Thread(target=self.update, args=(), daemon=True, name="CameraThread").start()
        return self

    def update(self):
        while not self.stopped.is_set():
            with self.lock:
                ret, frame = self.stream.read()
                if not ret:
                    logging.warning("Camera read failed, attempting recovery...")
                    self.recover_camera()
                    continue

                # Update FPS calculation
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    current_time = time.time()
                    self.fps = 30 / (current_time - self.fps_start_time)
                    self.fps_start_time = current_time

                # Maintain latest frame
                self.last_frame = frame
                self.frame_ready.set()

                # Update queue with frame
                if not self.q.full():
                    self.q.put(frame)
                else:
                    try:
                        self.q.get_nowait()  # Remove oldest frame
                        self.q.put(frame)    # Add new frame
                    except queue.Empty:
                        pass

    def recover_camera(self):
        """Attempt to recover camera connection"""
        try:
            self.stream.release()
            time.sleep(1)
            self.stream = cv2.VideoCapture(self.src)
            logging.info("Camera connection recovered")
        except Exception as e:
            logging.error(f"Camera recovery failed: {str(e)}")

    def read(self):
        """Non-blocking frame read"""
        return self.q.get() if not self.q.empty() else self.last_frame

    def read_latest(self):
        """Get the most recent frame"""
        while not self.frame_ready.is_set() and not self.stopped.is_set():
            time.sleep(0.01)
        return self.last_frame

    def get_fps(self):
        return self.fps

    def stop(self):
        self.stopped.set()
        with self.lock:
            self.stream.release()

class AlertSystem:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.sound_fire = pygame.mixer.Sound("assets/fire_alert.wav")
        self.sound_fall = pygame.mixer.Sound("assets/fall_alert.wav")
        self.alert_thread = Thread(target=self._alert_loop, daemon=True)
        self.alert_queue = queue.Queue()
        self.alert_thread.start()

    def _alert_loop(self):
        while True:
            alert_type = self.alert_queue.get()
            if alert_type == "fire":
                self.sound_fire.play()
            elif alert_type == "fall":
                self.sound_fall.play()
            time.sleep(2)  # Minimum delay between alerts

    def trigger_alert(self, alert_type):
        self.alert_queue.put(alert_type)

class IncidentRecorder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.incident_log = os.path.join(output_dir, "incident_log.json")
        self.load_incident_history()

    def load_incident_history(self):
        try:
            if os.path.exists(self.incident_log):
                with open(self.incident_log, 'r') as f:
                    self.history = json.load(f)
            else:
                self.history = []
        except Exception as e:
            logging.error(f"Failed to load incident history: {str(e)}")
            self.history = []

    def save_incident(self, frame, incident_type, confidence, metadata=None):
        timestamp = datetime.now()
        filename = f"{incident_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            
            incident_data = {
                "type": incident_type,
                "timestamp": timestamp.isoformat(),
                "filename": filename,
                "confidence": confidence,
                "metadata": metadata or {}
            }
            
            self.history.append(incident_data)
            
            with open(self.incident_log, 'w') as f:
                json.dump(self.history, f, indent=2)
                
            logging.info(f"Incident saved: {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save incident: {str(e)}")
            return False

class SmartCCTV:
    def __init__(self, camera_source="0"):
        # Initialize GPU acceleration if available
        self.gpu_available = tf.test.is_built_with_cuda()
        if self.gpu_available:
            logging.info("GPU acceleration enabled")
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Initialize MediaPipe with enhanced settings
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=2  # Using highest accuracy model
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize components
        self.camera_source = camera_source
        self.vs = None
        self.setup_camera()
        self.alert_system = AlertSystem()
        self.incident_recorder = IncidentRecorder("incidents")
        
        # Enhanced fall detection parameters
        self.fall_params = {
            'threshold': 0.25,
            'frames_threshold': 3,
            'counter': 0,
            'prev_ratio': None,
            'movement_threshold': 0.3,
            'confidence_history': [],
            'history_size': 10
        }
        
        # Enhanced fire detection parameters
        self.fire_params = {
            'lower1': np.array([0, 100, 100]),
            'upper1': np.array([25, 255, 255]),
            'lower2': np.array([160, 100, 100]),
            'upper2': np.array([180, 255, 255]),
            'min_area': 200,
            'max_area': 100000,
            'confidence': 0.0,
            'confidence_threshold': 0.4,
            'detection_history': [],
            'history_size': 5
        }
        
        # Performance optimization
        self.frame_processor = FrameProcessor()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.detection_results = {}
        self.last_incident_time = {'fall': 0, 'fire': 0}
        self.min_incident_interval = 3
        self.frame_count = 0
        self.process_every_n_frames = 2
        
        logging.info("Enhanced Smart CCTV System Initialized")

    def setup_camera(self):
        """Enhanced camera setup with recovery"""
        try:
            if self.vs is not None:
                self.vs.stop()
            
            src = int(self.camera_source) if self.camera_source.isdigit() else self.camera_source
            self.vs = VideoStream(src).start()
            time.sleep(2.0)
            
            test_frame = self.vs.read_latest()
            if test_frame is None:
                raise ValueError("Camera initialization failed")
                
        except Exception as e:
            logging.error(f"Camera setup failed: {str(e)}")
            raise

    def detect_fire(self, frame):
        """Enhanced fire detection with temporal analysis"""
        try:
            # Process frame
            small_frame = self.frame_processor.resize_frame(frame, 0.5)
            hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
            
            # Create and combine fire masks
            mask = self.frame_processor.create_fire_mask(
                hsv, 
                self.fire_params['lower1'], 
                self.fire_params['upper1'],
                self.fire_params['lower2'], 
                self.fire_params['upper2']
            )
            
            # Analyze fire characteristics
            fire_detected, confidence = self.frame_processor.analyze_fire(
                small_frame, 
                mask, 
                self.fire_params
            )
            
            # Update detection history
            self.fire_params['detection_history'].append(fire_detected)
            if len(self.fire_params['detection_history']) > self.fire_params['history_size']:
                self.fire_params['detection_history'].pop(0)
            
            # Temporal analysis
            if len(self.fire_params['detection_history']) >= 3:
                fire_detected = sum(self.fire_params['detection_history'][-3:]) >= 2
            
            # Update confidence and draw alert
            self.fire_params['confidence'] = confidence
            if fire_detected:
                self.frame_processor.draw_fire_alert(frame, confidence)
                self.alert_system.trigger_alert("fire")
            
            return fire_detected, confidence
            
        except Exception as e:
            logging.error(f"Fire detection error: {str(e)}")
            return False, 0.0

    def detect_fall(self, frame):
        """Enhanced fall detection with temporal analysis"""
        try:
            # Process pose detection
            results = self.frame_processor.detect_pose(frame, self.pose)
            if not results.pose_landmarks:
                return False, 0.0
            
            # Calculate fall metrics
            metrics = self.frame_processor.calculate_fall_metrics(results.pose_landmarks, self.mp_pose)
            
            # Detect fall and get confidence
            fall_detected, confidence = self.frame_processor.analyze_fall(
                metrics,
                self.fall_params,
                frame
            )
            
            # Update confidence history
            self.fall_params['confidence_history'].append(confidence)
            if len(self.fall_params['confidence_history']) > self.fall_params['history_size']:
                self.fall_params['confidence_history'].pop(0)
            
            # Temporal analysis
            if fall_detected and len(self.fall_params['confidence_history']) >= 3:
                avg_confidence = sum(self.fall_params['confidence_history'][-3:]) / 3
                fall_detected = avg_confidence > 0.6
            
            if fall_detected:
                self.alert_system.trigger_alert("fall")
            
            return fall_detected, confidence
            
        except Exception as e:
            logging.error(f"Fall detection error: {str(e)}")
            return False, 0.0

    def process_frame(self, frame):
        """Process frame with parallel detection"""
        try:
            display_frame = frame.copy()
            
            # Run detections in parallel
            fire_future = self.thread_pool.submit(self.detect_fire, display_frame)
            fall_future = self.thread_pool.submit(self.detect_fall, display_frame)
            
            # Get results
            fire_detected, fire_confidence = fire_future.result()
            fall_detected, fall_confidence = fall_future.result()
            
            # Save incidents with metadata
            if fire_detected:
                self.incident_recorder.save_incident(
                    display_frame, 
                    "fire", 
                    fire_confidence,
                    {"detection_history": self.fire_params['detection_history']}
                )
            
            if fall_detected:
                self.incident_recorder.save_incident(
                    display_frame, 
                    "fall", 
                    fall_confidence,
                    {"body_metrics": self.fall_params['confidence_history']}
                )
            
            # Add overlay information
            self.frame_processor.add_overlay(
                display_frame,
                self.camera_source,
                self.vs.get_fps(),
                fire_detected or fall_detected,
                {
                    "fire_confidence": fire_confidence,
                    "fall_confidence": fall_confidence,
                    "gpu_enabled": self.gpu_available
                }
            )
            
            return display_frame, fire_detected or fall_detected
            
        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")
            return frame, False

    def run(self):
        """Enhanced main loop with parallel processing"""
        try:
            while True:
                # Get latest frame
                frame = self.vs.read_latest()
                if frame is None:
                    logging.warning("Failed to grab frame, attempting to reconnect...")
                    self.setup_camera()
                    continue

                # Process frames at optimal rate
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames != 0:
                    self.frame_processor.display_frame('Smart CCTV', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Process frame and display results
                display_frame, incident_detected = self.process_frame(frame)
                self.frame_processor.display_frame('Smart CCTV', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logging.info("Gracefully shutting down...")
        except Exception as e:
            logging.error(f"Runtime error: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Enhanced cleanup"""
        if self.vs is not None:
            self.vs.stop()
        self.thread_pool.shutdown()
        cv2.destroyAllWindows()
        pygame.quit()
        logging.info("System shutdown complete")

class FrameProcessor:
    """Enhanced frame processing operations"""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.performance_stats = {
            'last_fps_update': time.time(),
            'fps_values': [],
            'avg_fps': 0
        }

    def resize_frame(self, frame, scale):
        return cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    def create_fire_mask(self, hsv, lower1, upper1, lower2, upper2):
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        combined = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((3,3), np.uint8)
        combined = cv2.erode(combined, kernel, iterations=1)
        combined = cv2.dilate(combined, kernel, iterations=2)
        
        return combined

    def analyze_fire(self, frame, mask, params):
        red_channel = frame[:,:,2]
        avg_intensity = np.mean(red_channel[mask > 0]) if np.sum(mask) > 0 else 0
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        confidence = params['confidence']
        fire_detected = False
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if params['min_area'] < area < params['max_area']:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                
                if 0.3 < aspect_ratio < 3.0 and avg_intensity > 120:
                    confidence = min(1.0, confidence + 0.3)
                    if confidence >= params['confidence_threshold']:
                        fire_detected = True
                        break
        
        if not fire_detected:
            confidence = max(0, confidence - 0.05)
        
        return fire_detected, confidence

    def detect_pose(self, frame, pose):
        return pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def calculate_fall_metrics(self, landmarks, mp_pose):
        # Get key points
        keypoints = {}
        for point in [
            'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_HIP', 'RIGHT_HIP', 'LEFT_ANKLE', 'RIGHT_ANKLE'
        ]:
            keypoints[point] = landmarks.landmark[getattr(mp_pose.PoseLandmark, point)]
        
        # Calculate averages
        shoulder_y = (keypoints['LEFT_SHOULDER'].y + keypoints['RIGHT_SHOULDER'].y) / 2
        hip_y = (keypoints['LEFT_HIP'].y + keypoints['RIGHT_HIP'].y) / 2
        ankle_y = (keypoints['LEFT_ANKLE'].y + keypoints['RIGHT_ANKLE'].y) / 2
        
        # Calculate body angle
        body_vector = np.array([hip_y - shoulder_y, 1.0])
        vertical_vector = np.array([0, 1.0])
        body_angle = np.degrees(np.arccos(np.dot(body_vector, vertical_vector) / 
                                        (np.linalg.norm(body_vector) * np.linalg.norm(vertical_vector))))
        
        return {
            'keypoints': keypoints,
            'shoulder_y': shoulder_y,
            'hip_y': hip_y,
            'ankle_y': ankle_y,
            'body_angle': body_angle,
            'current_ratio': hip_y / shoulder_y
        }

    def analyze_fall(self, metrics, params, frame):
        confidence = 0.0
        
        # Update sudden movement detection
        if params['prev_ratio'] is not None:
            sudden_movement = abs(metrics['current_ratio'] - params['prev_ratio'])
            if sudden_movement > params['movement_threshold']:
                params['counter'] += 1
                confidence += 0.3
        params['prev_ratio'] = metrics['current_ratio']
        
        # Check fall conditions
        horizontal_body = abs(metrics['hip_y'] - metrics['shoulder_y']) < params['threshold']
        head_below_hips = metrics['keypoints']['NOSE'].y > metrics['hip_y']
        large_angle = metrics['body_angle'] > 60
        
        if horizontal_body:
            confidence += 0.3
        if head_below_hips:
            confidence += 0.2
        if large_angle:
            confidence += 0.2
        
        fall_detected = False
        if (horizontal_body and head_below_hips) or large_angle:
            params['counter'] += 1
            if params['counter'] >= params['frames_threshold']:
                self.draw_fall_alert(frame, metrics['body_angle'], confidence)
                fall_detected = True
        else:
            params['counter'] = max(0, params['counter'] - 1)
        
        # Draw debug info
        self.draw_pose_debug(frame, metrics)
        
        return fall_detected, confidence

    def draw_pose_debug(self, frame, metrics):
        cv2.putText(frame, f'Body Angle: {metrics["body_angle"]:.1f}°', (10, 70),
                    self.font, 0.5, (255, 255, 255), 1, self.line_type)
        cv2.putText(frame, f'Hip/Shoulder: {metrics["current_ratio"]:.2f}', (10, 90),
                    self.font, 0.5, (255, 255, 255), 1, self.line_type)

    def draw_fire_alert(self, frame, confidence):
        cv2.putText(frame, f'FIRE DETECTED! ({confidence:.1f})',
                    (50, 50), self.font, 0.9, (0, 0, 255), 2, self.line_type)

    def draw_fall_alert(self, frame, angle, confidence):
        cv2.putText(frame, f'FALL DETECTED! (Angle: {angle:.1f}°, Conf: {confidence:.1f})',
                    (50, 50), self.font, 1, (0, 0, 255), 2, self.line_type)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 4)

    def add_overlay(self, frame, camera_source, fps, incident_detected, stats=None):
        # Add timestamp
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (10, frame.shape[0] - 10), self.font, 0.5, (0, 255, 0), 1, self.line_type)

        # Add enhanced status information
        y_offset = 30
        cv2.putText(frame, f"Camera: {camera_source} | FPS: {int(fps)}",
                    (10, y_offset), self.font, 0.5, (255, 255, 255), 1, self.line_type)
        
        if stats:
            y_offset += 20
            if stats.get('gpu_enabled'):
                cv2.putText(frame, "GPU Accelerated", (10, y_offset),
                           self.font, 0.5, (255, 255, 0), 1, self.line_type)
            
            y_offset += 20
            if 'fire_confidence' in stats:
                cv2.putText(frame, f"Fire Conf: {stats['fire_confidence']:.2f}",
                           (10, y_offset), self.font, 0.5, (255, 255, 255), 1, self.line_type)
            
            y_offset += 20
            if 'fall_confidence' in stats:
                cv2.putText(frame, f"Fall Conf: {stats['fall_confidence']:.2f}",
                           (10, y_offset), self.font, 0.5, (255, 255, 255), 1, self.line_type)

        # Add status indicator with pulse effect
        if incident_detected:
            pulse = int(time.time() * 4) % 2  # Create pulsing effect
            radius = 10 + pulse * 2
            cv2.circle(frame, (20, 20), radius, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (20, 20), 10, (0, 255, 0), -1)

    def display_frame(self, window_name, frame):
        cv2.imshow(window_name, frame)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Smart CCTV System')
    parser.add_argument('--camera', default='0',
                      help='Camera source (0 for webcam, or IP camera URL)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with additional visualizations')
    args = parser.parse_args()

    try:
        cctv = SmartCCTV(camera_source=args.camera)
        cctv.run()
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 