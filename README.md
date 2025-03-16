# Smart CCTV System with Fall and Fire Detection

This project implements an AI-powered CCTV system that can detect falls and fire incidents in real-time using computer vision and machine learning techniques.

## Features

- Real-time fall detection using pose estimation
- Fire detection using color thresholding
- Incident logging with timestamps
- Automatic saving of detected incidents
- Live video feed with detection overlays

## Requirements

- Python 3.8 or higher
- Webcam or USB camera
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:
   ```bash
   python smart_cctv.py
   ```

2. The system will:
   - Start your camera
   - Begin monitoring for falls and fire
   - Display the live feed with detection overlays
   - Save incidents to the 'incidents' folder when detected

3. Press 'q' to quit the application

## How it Works

### Fall Detection
- Uses MediaPipe Pose Detection to track body landmarks
- Analyzes the relative positions of key points (hip and shoulder)
- Triggers when unusual posture is detected

### Fire Detection
- Uses color thresholding in HSV color space
- Detects flame-like colors and patterns
- Triggers when fire-colored regions exceed size threshold

## Output
- Detected incidents are saved in the 'incidents' folder
- Each incident is saved with timestamp and type (fall/fire)
- Live feed shows detection boxes and warnings

## Notes
- Adjust `fall_threshold` in the code to fine-tune fall detection sensitivity
- Modify fire detection thresholds if needed for different lighting conditions
- Ensure good lighting for optimal detection 