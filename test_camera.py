import cv2
import time

def test_camera():
    print("Attempting to access camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    print("Camera opened successfully")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print("Successfully read a frame")
        print(f"Frame shape: {frame.shape}")
    else:
        print("Failed to read frame")
    
    # Release the camera
    cap.release()

if __name__ == "__main__":
    test_camera() 