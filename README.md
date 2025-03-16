# Smart CCTV System with Fall and Fire Detection

This project implements an AI-powered CCTV system that can detect falls and fire incidents in real-time using computer vision and machine learning techniques.

## Features

- Real-time fall detection using advanced pose estimation and biomechanical analysis
- Fire detection using multi-threshold HSV color analysis and region tracking
- Audio alerts for detected incidents
- Comprehensive incident logging with timestamps and confidence scores
- Automatic saving of detected incidents with metadata
- Live video feed with detection overlays and performance stats
- Multi-threaded design for optimal performance
- Automatic camera recovery in case of disconnection

## Requirements

- Python 3.8 or higher
- Webcam or USB camera
- Required Python packages (listed in requirements.txt)
- Audio output device for alerts

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:
   ```bash
   python smart_cctv.py [options]
   ```

   Available options:
   - `--camera`: Camera source (default: 0)
   - `--resolution`: Video resolution (default: 1280x720)
   - `--fps`: Target FPS (default: 30)

2. The system will:
   - Start your camera with optimized settings
   - Begin monitoring for falls and fire
   - Display the live feed with detection overlays and performance stats
   - Play audio alerts when incidents are detected
   - Save incidents to the 'incidents' folder
   - Log all events to 'cctv.log'

3. Press 'q' to quit the application

## How it Works

### Fall Detection
- Uses MediaPipe Pose Detection to track 33 body landmarks
- Analyzes multiple biomechanical metrics:
  - Vertical displacement of key points
  - Body angle relative to ground
  - Movement velocity
  - Pose stability
- Uses confidence thresholds to minimize false positives

### Fire Detection
- Multi-stage HSV color space analysis
- Dual-threshold fire color detection
- Region size and shape analysis
- Temporal consistency checking
- Confidence scoring based on multiple parameters

### Alert System
- Real-time audio alerts using pygame
- Different alert sounds for different incident types
- Visual overlays with incident type and confidence
- Timestamp and performance statistics display

## Output
- Detected incidents are saved in the 'incidents' folder with:
  - Timestamped image files
  - JSON metadata including:
    - Incident type
    - Confidence score
    - Detection parameters
    - Camera settings
- Comprehensive logging in 'cctv.log' with:
  - System events
  - Detection events
  - Performance metrics
  - Error tracking

## Notes
- The system uses GPU acceleration when available
- Camera settings are automatically optimized for detection
- Multiple fail-safes for camera disconnection and recovery
- Configurable detection parameters in the code
- Thread-safe design for reliable operation 