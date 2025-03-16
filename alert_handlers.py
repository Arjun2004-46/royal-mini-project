import logging
import pygame
import queue
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self):
        self.use_sound = True
        
        # Initialize callback registry for external alert handlers
        self.alert_callbacks = {}
        
        if self.use_sound:
            pygame.init()
            pygame.mixer.init()
            self.sound_fire = pygame.mixer.Sound("assets/fire_alert.wav")
            self.sound_fall = pygame.mixer.Sound("assets/fall_alert.wav")
            
            # Set default volume
            volume = 1.0
            self.sound_fire.set_volume(volume)
            self.sound_fall.set_volume(volume)
            
            self.alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
            self.alert_queue = queue.Queue()
            self.alert_thread.start()
            logger.info("Sound alerts enabled")
        else:
            logger.info("Sound alerts disabled")

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
            time.sleep(2.0)  # Minimum delay between alerts

    def register_alert_callback(self, alert_type, callback):
        """Register a callback function for a specific alert type"""
        if alert_type not in self.alert_callbacks:
            self.alert_callbacks[alert_type] = []
        self.alert_callbacks[alert_type].append(callback)
        logger.info(f"Registered new alert callback for {alert_type}")
        
    def trigger_alert(self, alert_type, metadata=None):
        """Trigger an alert with optional metadata"""
        # Play sound alert
        if self.use_sound:
            self.alert_queue.put(alert_type)
            
        # Call registered callbacks
        if alert_type in self.alert_callbacks:
            for callback in self.alert_callbacks[alert_type]:
                try:
                    callback(alert_type, metadata)
                except Exception as e:
                    logger.error(f"Error in alert callback: {str(e)}")

    def cleanup(self):
        """Cleanup resources"""
        if self.use_sound:
            pygame.quit() 