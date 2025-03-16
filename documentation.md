# Smart CCTV System with Fall and Fire Detection

## Project Overview
This is an advanced AI-powered CCTV system that leverages computer vision and machine learning to detect falls and fire incidents in real-time. The project demonstrates practical applications of artificial intelligence in safety and surveillance systems.

## Architecture

### Core Components
1. **Smart CCTV Core (`smart_cctv.py`)**
   - Main detection engine
   - Handles real-time video processing
   - Implements fall and fire detection algorithms
   - Manages incident logging and alerts

2. **Web Interface**
   - **Web Server (`web_server.py`)**: Handles HTTP requests and serves web interface
   - **API Server (`api_server.py`)**: Provides RESTful API endpoints
   - **Streaming Server (`streaming_server.py`)**: Manages real-time video streaming
   - **Web Client (`web-client/`)**: Frontend interface for monitoring and control

3. **Support Components**
   - **Configuration (`config.json`)**: System settings and parameters
   - **Startup Script (`start.sh`)**: Automated deployment script
   - **Camera Test (`test_camera.py`)**: Camera setup and testing utility

## Key Features

### 1. Intelligent Detection
- **Fall Detection**
  - Uses MediaPipe for pose estimation
  - Analyzes body posture and movement
  - Real-time biomechanical analysis
  - Multi-parameter confidence scoring

- **Fire Detection**
  - HSV color space analysis
  - Multi-threshold detection
  - Region tracking and validation
  - Temporal consistency checks

### 2. Real-Time Monitoring
- Live video feed with detection overlays
- Performance statistics display
- Audio alerts for incidents
- Automatic camera recovery

### 3. Incident Management
- Automatic incident logging
- Timestamped incident captures
- Detailed metadata storage
- Comprehensive event logging

## Technical Stack

### Dependencies
- **Core Libraries**
  - OpenCV (≥4.5.0): Computer vision operations
  - TensorFlow (≥2.5.0): Machine learning backend
  - MediaPipe (≥0.8.0): Pose estimation
  - NumPy (≥1.19.0): Numerical computations

- **Web Stack**
  - Flask (≥2.0.0): Web server framework
  - Flask-SocketIO (≥5.0.0): Real-time communication
  - Flask-CORS (≥3.0.0): Cross-origin resource sharing
  - Eventlet (≥0.30.0): Async networking

### System Requirements
- Python 3.8 or higher
- Compatible webcam or USB camera
- Audio output capability
- Sufficient processing power for real-time analysis

## Project Structure
```
.
├── smart_cctv.py         # Main detection engine
├── api_server.py         # REST API implementation
├── streaming_server.py   # Video streaming service
├── web_server.py        # Web interface server
├── config.json          # System configuration
├── start.sh            # Deployment script
├── requirements.txt    # Python dependencies
├── incidents/         # Incident storage
├── web-client/       # Frontend application
└── assets/          # Project resources
```

## Setup and Deployment

1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**
   - Adjust settings in `config.json`
   - Configure camera parameters
   - Set detection thresholds

3. **Running the System**
   ```bash
   ./start.sh
   ```
   or
   ```bash
   python smart_cctv.py
   ```

## Output and Logging

### Incident Records
- Stored in `incidents/` directory
- Includes:
  - Timestamped images
  - Incident metadata
  - Detection parameters
  - Camera settings

### System Logs
- Main log file: `cctv.log`
- Contains:
  - System events
  - Detection results
  - Performance metrics
  - Error records

## Future Enhancements
1. Integration with mobile notifications
2. Cloud storage support
3. Multi-camera support
4. Advanced analytics dashboard
5. Machine learning model improvements

This project demonstrates practical applications of computer vision and machine learning in real-world safety systems, making it an excellent educational example of modern AI applications. 