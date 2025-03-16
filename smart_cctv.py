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
import uuid

# Load configuration
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("config.json not found. Using default configuration.")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config.json: {str(e)}")
        return None

CONFIG = load_config()

# Configure logging with more detailed format
logging.basicConfig(
    level=getattr(logging, CONFIG.get('LOGGING', {}).get('LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG.get('LOGGING', {}).get('FILE', 'cctv.log')),
        logging.StreamHandler()
    ]
)

class VideoStream:
    def __init__(self, src=0):
        # Initialize camera with AVFoundation on macOS
        self.stream = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        self.src = src
        
        # Get camera settings from config
        camera_config = CONFIG.get('CAMERA', {})
        
        # Set basic camera properties first
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get('WIDTH', 1280))
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get('HEIGHT', 720))
        self.stream.set(cv2.CAP_PROP_FPS, camera_config.get('FPS', 30))
        
        # Advanced threading
        self.q = queue.Queue(maxsize=4)
        self.lock = Lock()
        self.stopped = Event()
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        self.last_frame = None
        self.frame_ready = Event()

        # Check if camera is opened successfully
        if not self.stream.isOpened():
            raise ValueError("Error: Could not open camera")
            
        # Verify we can read a frame
        ret, frame = self.stream.read()
        if not ret or frame is None:
            raise ValueError("Error: Could not read frame from camera")
            
        logging.info(f"Camera initialized successfully - Resolution: {frame.shape[1]}x{frame.shape[0]}, FPS: {self.stream.get(cv2.CAP_PROP_FPS)}")

    def start(self):
        """Start the video stream thread"""
        Thread(target=self.update, args=(), daemon=True, name="CameraThread").start()
        return self

    def update(self):
        """Update the video stream"""
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
                self.last_frame = frame.copy()  # Make a copy to avoid race conditions
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

    def read_latest(self):
        """Read the most recent frame"""
        if self.last_frame is not None:
            with self.lock:
                return self.last_frame.copy()
        return None

    def read(self):
        """Read the next frame from the queue"""
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None

    def get_fps(self):
        """Get the current FPS of the video stream"""
        return self.fps

    def recover_camera(self):
        """Attempt to recover the camera connection"""
        with self.lock:
            self.stream.release()
            time.sleep(1.0)
            self.stream = cv2.VideoCapture(self.src)
            if not self.stream.isOpened():
                logging.error("Failed to recover camera connection")
                self.stop()

    def stop(self):
        """Stop the video stream"""
        self.stopped.set()
        with self.lock:
            if self.stream is not None:
                self.stream.release()
        logging.info("Video stream stopped")

class AlertSystem:
    def __init__(self):
        self.config = CONFIG.get('ALERTS', {}).get('SOUND', {})
        self.use_sound = self.config.get('ENABLED', True) and CONFIG.get('USE_SOUND', True)
        
        if self.use_sound:
            pygame.init()
            pygame.mixer.init()
            self.sound_fire = pygame.mixer.Sound("assets/fire_alert.wav")
            self.sound_fall = pygame.mixer.Sound("assets/fall_alert.wav")
            
            # Set volume from config
            volume = self.config.get('VOLUME', 1.0)
            self.sound_fire.set_volume(volume)
            self.sound_fall.set_volume(volume)
            
            self.alert_thread = Thread(target=self._alert_loop, daemon=True)
            self.alert_queue = queue.Queue()
            self.alert_thread.start()
            logging.info("Sound alerts enabled")
        else:
            logging.info("Sound alerts disabled")

    def _alert_loop(self):
        while True:
            if not self.use_sound:
                time.sleep(1)
                continue
                
            alert_type = self.alert_queue.get()
            if alert_type == "fire":
                self.sound_fire.play()
            elif alert_type == "fall":
                self.sound_fall.play()
            time.sleep(self.config.get('MIN_DELAY', 2.0))

    def trigger_alert(self, alert_type):
        if self.use_sound:
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
                    # Add UUIDs to existing incidents if they don't have one
                    modified = False
                    for incident in self.history:
                        if 'uuid' not in incident:
                            incident['uuid'] = str(uuid.uuid4())
                            modified = True
                    if modified:
                        with open(self.incident_log, 'w') as f:
                            json.dump(self.history, f, indent=2)
            else:
                self.history = []
        except Exception as e:
            logging.error(f"Failed to load incident history: {str(e)}")
            self.history = []

    def save_incident(self, frame, incident_type, confidence, metadata=None):
        timestamp = datetime.now()
        incident_uuid = str(uuid.uuid4())
        filename = f"{incident_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{incident_uuid[:8]}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            
            incident_data = {
                "uuid": incident_uuid,
                "type": incident_type,
                "timestamp": timestamp.isoformat(),
                "image": filename,  # Changed from filename to image for consistency
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
        # Store fire parameters as instance variables
        self.fire_params = None

    def set_fire_params(self, params):
        """Set fire detection parameters"""
        self.fire_params = params

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
        """Enhanced fire detection with more sensitive parameters"""
        self.fire_params = params
        
        if params['prev_frame'] is not None:
            frame_diff = cv2.absdiff(frame, params['prev_frame'])
            movement = np.mean(frame_diff)
            
            if movement > params['movement_threshold']:
                current_time = time.time()
                params['last_movement_time'] = current_time
                
            if time.time() - params['last_movement_time'] < 0.5:  # Reduced cooldown time
                params['confidence'] = max(0, params['confidence'] - 0.1)  # Less confidence reduction
                
        params['prev_frame'] = frame.copy()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(frame)
        
        red_intensity = np.mean(r[mask > 0]) if np.sum(mask) > 0 else 0
        other_intensity = np.mean((b[mask > 0] + g[mask > 0]) / 2) if np.sum(mask) > 0 else 0
        
        params['prev_intensities'].append(red_intensity)
        if len(params['prev_intensities']) > params['intensity_history_size']:
            params['prev_intensities'].pop(0)
        
        intensity_variation = np.std(params['prev_intensities']) / (np.mean(params['prev_intensities']) + 1e-6)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        confidence = params['confidence']
        fire_detected = False
        detected_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if params['min_area'] < area < params['max_area']:
                x, y, w, h = cv2.boundingRect(contour)
                roi_hsv = hsv[y:y+h, x:x+w]
                
                aspect_ratio = float(w)/h
                saturation = np.mean(roi_hsv[:,:,1])
                intensity_ratio = red_intensity / (other_intensity + 1e-6)
                
                if (0.3 < aspect_ratio < 3.0 and  # More lenient aspect ratio
                    saturation > params['min_saturation'] and
                    intensity_ratio > params['min_intensity_ratio']):
                    
                    ratio_confidence = min(intensity_ratio / 1.5, 0.5)  # More lenient ratio scoring
                    saturation_confidence = min((saturation - 100) / 100.0, 0.5)  # More lenient saturation scoring
                    flicker_confidence = min(intensity_variation * 3, 0.3)  # Enhanced flicker importance
                    
                    frame_confidence = (ratio_confidence * 0.4 + 
                                     saturation_confidence * 0.4 + 
                                     flicker_confidence * 0.2)
                    
                    confidence = min(1.0, confidence + frame_confidence)
                    
                    if confidence >= params['confidence_threshold']:
                        detected_regions.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'intensity_ratio': intensity_ratio,
                            'saturation': saturation
                        })
                        fire_detected = True
        
        # Draw all detected regions with detailed information
        for region in detected_regions:
            x, y, w, h = region['bbox']
            
            # Draw filled contour with semi-transparency
            overlay = frame.copy()
            cv2.drawContours(overlay, [region['contour']], -1, (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw prominent bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
            # Add detailed information with better visibility
            info_text = [
                f"FIRE DETECTED",
                f"Confidence: {region['confidence']:.2f}",
                f"Int Ratio: {region['intensity_ratio']:.2f}",
                f"Sat: {region['saturation']:.0f}"
            ]
            
            # Draw text background for better visibility
            for i, text in enumerate(info_text):
                text_size = cv2.getTextSize(text, self.font, 0.7, 2)[0]
                cv2.rectangle(frame, 
                            (x, y - 30 - (i * 25)), 
                            (x + text_size[0], y - 10 - (i * 25)), 
                            (0, 0, 0), 
                            -1)
                cv2.putText(frame, text, 
                           (x, y - 15 - (i * 25)),
                           self.font, 0.7, (255, 255, 255), 2, self.line_type)
        
        if not fire_detected:
            confidence = max(0, confidence - 0.02)  # Slower confidence decay
        
        return fire_detected, confidence

    def detect_pose(self, frame, pose):
        """Enhanced pose detection with RGB conversion"""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame and return results
        return pose.process(rgb_frame)

    def draw_pose_landmarks(self, frame, results, mp_draw, mp_pose, landmark_spec, connection_spec):
        """Draw pose landmarks with custom visualization"""
        if results.pose_landmarks:
            # Draw the pose landmarks
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_spec,
                connection_spec
            )
            
            # Add visibility scores for key points
            visible_points = 0
            total_points = 33  # Total number of landmarks
            
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if landmark.visibility > 0.5:  # Consider point visible if visibility > 50%
                    visible_points += 1
            
            visibility_percentage = (visible_points / total_points) * 100
            
            # Draw visibility information
            cv2.putText(frame, f'Body Visibility: {visibility_percentage:.1f}%',
                       (10, 30), self.font, 0.5, (0, 255, 0), 1, self.line_type)

    def calculate_fall_metrics(self, landmarks, mp_pose):
        """Enhanced fall metrics calculation with partial detection support"""
        keypoints = {}
        visible_keypoints = {}
        
        # Key points to track with their respective importance for fall detection
        key_points = {
            'NOSE': 0.5,
            'LEFT_SHOULDER': 1.0,
            'RIGHT_SHOULDER': 1.0,
            'LEFT_HIP': 1.0,
            'RIGHT_HIP': 1.0,
            'LEFT_ANKLE': 0.7,
            'RIGHT_ANKLE': 0.7
        }
        
        # Get available keypoints and their visibility
        for point, importance in key_points.items():
            landmark = landmarks.landmark[getattr(mp_pose.PoseLandmark, point)]
            keypoints[point] = landmark
            visible_keypoints[point] = landmark.visibility > 0.5
        
        # Calculate metrics based on available points
        metrics = {
            'keypoints': keypoints,
            'visible_points': visible_keypoints,
            'body_angle': 0,
            'current_ratio': 1.0
        }
        
        # Calculate shoulder and hip positions if available
        if visible_keypoints['LEFT_SHOULDER'] and visible_keypoints['RIGHT_SHOULDER']:
            metrics['shoulder_y'] = (keypoints['LEFT_SHOULDER'].y + keypoints['RIGHT_SHOULDER'].y) / 2
        elif visible_keypoints['LEFT_SHOULDER']:
            metrics['shoulder_y'] = keypoints['LEFT_SHOULDER'].y
        elif visible_keypoints['RIGHT_SHOULDER']:
            metrics['shoulder_y'] = keypoints['RIGHT_SHOULDER'].y
        
        if visible_keypoints['LEFT_HIP'] and visible_keypoints['RIGHT_HIP']:
            metrics['hip_y'] = (keypoints['LEFT_HIP'].y + keypoints['RIGHT_HIP'].y) / 2
        elif visible_keypoints['LEFT_HIP']:
            metrics['hip_y'] = keypoints['LEFT_HIP'].y
        elif visible_keypoints['RIGHT_HIP']:
            metrics['hip_y'] = keypoints['RIGHT_HIP'].y
        
        # Calculate body angle if we have both shoulder and hip positions
        if 'shoulder_y' in metrics and 'hip_y' in metrics:
            body_vector = np.array([metrics['hip_y'] - metrics['shoulder_y'], 1.0])
            vertical_vector = np.array([0, 1.0])
            metrics['body_angle'] = np.degrees(np.arccos(np.dot(body_vector, vertical_vector) / 
                                            (np.linalg.norm(body_vector) * np.linalg.norm(vertical_vector))))
            metrics['current_ratio'] = metrics['hip_y'] / metrics['shoulder_y']
        
        return metrics

    def analyze_fall(self, metrics, params, frame):
        """Enhanced fall analysis with more sensitive detection"""
        confidence = 0.0
        visible_points = metrics['visible_points']
        
        # Reduced minimum required points
        min_required_points = 3  # Reduced from 4 to 3
        visible_count = sum(1 for v in visible_points.values() if v)
        
        if visible_count < min_required_points:
            return False, 0.0
        
        # More sensitive vertical orientation scoring
        if 'body_angle' in metrics:
            angle_score = min(1.0, metrics['body_angle'] / params['vertical_threshold'])
            confidence += angle_score * 0.5  # Increased weight to 50%
        
        # More sensitive movement detection
        if params['prev_ratio'] is not None and 'current_ratio' in metrics:
            sudden_movement = abs(metrics['current_ratio'] - params['prev_ratio'])
            if sudden_movement > params['movement_threshold']:
                movement_score = min(1.0, sudden_movement / 0.3)  # More sensitive movement scoring
                confidence += movement_score * 0.3
        
        params['prev_ratio'] = metrics.get('current_ratio', params['prev_ratio'])
        
        # More lenient body position check
        if 'shoulder_y' in metrics and 'hip_y' in metrics:
            height_ratio = abs(metrics['shoulder_y'] - metrics['hip_y'])
            if height_ratio < params['height_ratio_threshold']:
                confidence += 0.3  # Increased position weight
        
        # More sensitive head position check
        if visible_points.get('NOSE', False) and 'hip_y' in metrics:
            head_below_hips = metrics['keypoints']['NOSE'].y > metrics['hip_y']
            if head_below_hips:
                confidence += 0.2  # Increased head position weight
        
        fall_detected = False
        if confidence > params['min_confidence']:
            params['counter'] += 2  # Faster counter increment
            if params['counter'] >= params['frames_threshold']:
                self.draw_fall_alert(frame, metrics.get('body_angle', 0), confidence)
                fall_detected = True
                self.draw_fall_debug(frame, metrics, confidence, params)
        else:
            params['counter'] = max(0, params['counter'] - 1)
        
        return fall_detected, confidence

    def draw_fall_debug(self, frame, metrics, confidence, params):
        """Enhanced fall detection visualization with detection area marking"""
        y_offset = frame.shape[0] - 140
        
        # Draw detection area
        if metrics.get('keypoints'):
            # Calculate bounding box around detected person
            x_coords = []
            y_coords = []
            for point, landmark in metrics['keypoints'].items():
                if metrics['visible_points'].get(point, False):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    x_coords.append(x)
                    y_coords.append(y)
            
            if x_coords and y_coords:
                # Add padding to bounding box
                padding = 20
                min_x = max(0, min(x_coords) - padding)
                min_y = max(0, min(y_coords) - padding)
                max_x = min(frame.shape[1], max(x_coords) + padding)
                max_y = min(frame.shape[0], max(y_coords) + padding)
                
                # Draw semi-transparent overlay for fall detection area
                overlay = frame.copy()
                cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), (255, 0, 0), -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                # Draw bounding box
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
                
                # Add fall detection information near the box
                info_text = [
                    f"Fall Conf: {confidence:.2f}",
                    f"Body Angle: {metrics.get('body_angle', 0):.1f}°"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(frame, text, (min_x, min_y - 10 - (i * 20)),
                              self.font, 0.5, (255, 0, 0), 2, self.line_type)
        
        # Show confidence breakdown
        cv2.putText(frame, f'Fall Detection Confidence: {confidence:.2f}', 
                   (10, y_offset), self.font, 0.6, (255, 0, 0), 2, self.line_type)
        y_offset -= 25
        
        if 'body_angle' in metrics:
            angle_text = f'Body Angle: {metrics["body_angle"]:.1f}° (Threshold: {params["vertical_threshold"]}°)'
            cv2.putText(frame, angle_text, 
                       (10, y_offset), self.font, 0.5, (255, 255, 255), 1, self.line_type)
            y_offset -= 20
        
        if 'current_ratio' in metrics:
            movement_text = f'Movement Score: {abs(metrics["current_ratio"] - params.get("prev_ratio", 0)):.2f}'
            cv2.putText(frame, movement_text,
                       (10, y_offset), self.font, 0.5, (255, 255, 255), 1, self.line_type)
            y_offset -= 20
        
        stability_text = f'Detection Frames: {params["counter"]}/{params["frames_threshold"]}'
        cv2.putText(frame, stability_text,
                   (10, y_offset), self.font, 0.5, (255, 255, 255), 1, self.line_type)
        
        # Draw body keypoints with confidence colors and labels
        if metrics.get('keypoints'):
            for point, landmark in metrics['keypoints'].items():
                if metrics['visible_points'].get(point, False):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    
                    # Draw point with visibility-based color
                    confidence_color = (0, int(255 * landmark.visibility), 0)
                    cv2.circle(frame, (x, y), 4, confidence_color, -1)
                    
                    # Draw point label with visibility score
                    cv2.putText(frame, f'{point}: {landmark.visibility:.2f}',
                              (x + 5, y), self.font, 0.4, confidence_color, 1, self.line_type)
                    
                    # Draw connecting lines between related points
                    if point in ['LEFT_SHOULDER', 'RIGHT_SHOULDER'] and 'NOSE' in metrics['keypoints']:
                        nose = metrics['keypoints']['NOSE']
                        nose_x = int(nose.x * frame.shape[1])
                        nose_y = int(nose.y * frame.shape[0])
                        cv2.line(frame, (x, y), (nose_x, nose_y), confidence_color, 1)

    def draw_fall_alert(self, frame, angle, confidence):
        """Enhanced fall alert visualization"""
        # Draw attention-grabbing alert box
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 4)
        
        # Create semi-transparent overlay for alert
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw alert text with multiple information levels
        cv2.putText(frame, f'FALL DETECTED!', 
                   (50, 35), self.font, 1.2, (255, 0, 0), 3, self.line_type)
        cv2.putText(frame, f'Angle: {angle:.1f}° | Confidence: {confidence:.2f}',
                   (50, 55), self.font, 0.7, (255, 255, 255), 2, self.line_type)
        
        # Add pulsing effect
        pulse = int(time.time() * 4) % 2
        if pulse:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 6)

    def draw_fire_alert(self, frame, confidence):
        """Enhanced fire alert visualization with debug info"""
        if self.fire_params and confidence >= self.fire_params['confidence_threshold']:
            # Draw attention-grabbing alert box
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 4)
            
            # Create semi-transparent overlay for alert
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Draw main alert text
            cv2.putText(frame, f'FIRE DETECTED!', 
                       (50, 35), self.font, 1.2, (0, 0, 255), 3, self.line_type)
            cv2.putText(frame, f'Confidence: {confidence:.2f}',
                       (50, 55), self.font, 0.7, (255, 255, 255), 2, self.line_type)
            
            # Add debug information
            y_offset = 90
            cv2.putText(frame, f'Threshold: {self.fire_params["confidence_threshold"]:.2f}',
                       (10, y_offset), self.font, 0.5, (255, 255, 255), 1, self.line_type)
            
            y_offset += 20
            cv2.putText(frame, f'Min Saturation: {self.fire_params["min_saturation"]}',
                       (10, y_offset), self.font, 0.5, (255, 255, 255), 1, self.line_type)
            
            y_offset += 20
            cv2.putText(frame, f'Min Intensity Ratio: {self.fire_params["min_intensity_ratio"]:.2f}',
                       (10, y_offset), self.font, 0.5, (255, 255, 255), 1, self.line_type)
            
            # Add pulsing effect
            pulse = int(time.time() * 4) % 2
            if pulse:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 6)

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

class SmartCCTV:
    def __init__(self, camera_source="0"):
        # Initialize GPU acceleration if available
        self.gpu_available = tf.test.is_built_with_cuda()
        if self.gpu_available:
            logging.info("GPU acceleration enabled")
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load configuration
        self.config = CONFIG or {}
        
        # Initialize MediaPipe with more lenient settings for partial detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.2,    # Lowered from 0.3
            min_tracking_confidence=0.2,     # Lowered from 0.3
            model_complexity=2,              # Keep highest accuracy model
            enable_segmentation=True,        # Enable segmentation for better partial detection
            smooth_landmarks=True            # Enable landmark smoothing
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Custom drawing specs for better visibility
        self.pose_landmark_drawing_spec = self.mp_draw.DrawingSpec(
            color=(0, 255, 0),  # Green color
            thickness=2,
            circle_radius=2
        )
        self.pose_connection_drawing_spec = self.mp_draw.DrawingSpec(
            color=(255, 255, 0),  # Yellow color
            thickness=2
        )
        
        # Initialize components
        self.camera_source = camera_source
        self.vs = None
        self.setup_camera()
        self.alert_system = AlertSystem()
        self.incident_recorder = IncidentRecorder("incidents")
        
        # Performance optimization
        self.frame_processor = FrameProcessor()
        
        # Load fall detection parameters from config
        fall_config = self.config.get('FALL_DETECTION', {})
        self.fall_params = {
            'threshold': fall_config.get('THRESHOLD', 0.05),
            'frames_threshold': fall_config.get('FRAMES_THRESHOLD', 2),
            'counter': 0,
            'prev_ratio': None,
            'movement_threshold': fall_config.get('MOVEMENT_THRESHOLD', 0.1),
            'confidence_history': [],
            'history_size': fall_config.get('HISTORY_SIZE', 3),
            'min_confidence': fall_config.get('MIN_CONFIDENCE', 0.2),
            'recovery_frames': fall_config.get('RECOVERY_FRAMES', 5),
            'vertical_threshold': fall_config.get('VERTICAL_THRESHOLD', 45),
            'height_ratio_threshold': fall_config.get('HEIGHT_RATIO_THRESHOLD', 0.3)
        }
        
        # Load fire detection parameters from config
        fire_config = self.config.get('FIRE_DETECTION', {})
        hue_ranges = fire_config.get('HUE_RANGES', [
            {'LOW': [0, 120, 120], 'HIGH': [25, 255, 255]},
            {'LOW': [160, 120, 120], 'HIGH': [179, 255, 255]}
        ])
        
        self.fire_params = {
            'lower1': np.array(hue_ranges[0]['LOW']),
            'upper1': np.array(hue_ranges[0]['HIGH']),
            'lower2': np.array(hue_ranges[1]['LOW']),
            'upper2': np.array(hue_ranges[1]['HIGH']),
            'min_area': fire_config.get('MIN_AREA', 100),
            'max_area': fire_config.get('MAX_AREA', 100000),
            'confidence': 0.0,
            'confidence_threshold': fire_config.get('CONFIDENCE_THRESHOLD', 0.4),
            'detection_history': [],
            'history_size': fire_config.get('HISTORY_SIZE', 5),
            'min_saturation': fire_config.get('MIN_SATURATION', 120),
            'min_intensity_ratio': fire_config.get('MIN_INTENSITY_RATIO', 1.2),
            'flicker_threshold': fire_config.get('FLICKER_THRESHOLD', 0.1),
            'prev_intensities': [],
            'intensity_history_size': fire_config.get('HISTORY_SIZE', 5),
            'movement_threshold': fire_config.get('MOVEMENT_THRESHOLD', 20.0),
            'prev_frame': None,
            'last_movement_time': 0
        }
        
        # Pass fire parameters to frame processor
        self.frame_processor.set_fire_params(self.fire_params)
        
        # Thread pool and other parameters from config
        detection_config = self.config.get('DETECTION', {})
        self.thread_pool = ThreadPoolExecutor(max_workers=detection_config.get('MAX_WORKERS', 4))
        self.detection_results = {}
        self.last_incident_time = {'fall': 0, 'fire': 0}
        self.min_incident_interval = detection_config.get('MIN_INCIDENT_INTERVAL', 3)
        self.frame_count = 0
        self.process_every_n_frames = detection_config.get('PROCESS_EVERY_N_FRAMES', 2)
        
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
        """Enhanced fall detection with partial body detection support"""
        try:
            # Process pose detection
            results = self.frame_processor.detect_pose(frame, self.pose)
            
            # Draw pose landmarks even if no fall is detected
            self.frame_processor.draw_pose_landmarks(frame, results, self.mp_draw, 
                                                   self.mp_pose, self.pose_landmark_drawing_spec,
                                                   self.pose_connection_drawing_spec)
            
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
        """Process frame with improved detection logic"""
        try:
            display_frame = frame.copy()
            
            fire_future = self.thread_pool.submit(self.detect_fire, display_frame)
            fall_future = self.thread_pool.submit(self.detect_fall, display_frame)
            
            fire_detected, fire_confidence = fire_future.result()
            fall_detected, fall_confidence = fall_future.result()
            
            # Modified mutual exclusion with less strict conditions
            if fall_detected and fall_confidence > 0.8:  # Only suppress fire detection at very high fall confidence
                fire_detected = False
                fire_confidence = 0.0
                self.fire_params['confidence'] = 0.0
            
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