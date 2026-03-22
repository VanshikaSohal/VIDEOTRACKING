import cv2
import numpy as np
import torch
import sys
from pathlib import Path

# Add YOLOv7 to Python path
yolov7_path = Path(__file__).parent / "yolov7"
sys.path.insert(0, str(yolov7_path))

print(f"Loading YOLOv7 from: {yolov7_path}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Import YOLOv7 modules
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

# Configuration
WEIGHTS_PATH = yolov7_path / "yolov7.pt"
VIDEO_PATH = "realvideo.mp4"
CONFIDENCE = 0.5
IOU_THRESHOLD = 0.45

# Check if weights exist
if not WEIGHTS_PATH.exists():
    print(f"ERROR: Weights not found at {WEIGHTS_PATH}")
    print("Please download yolov7.pt to the yolov7 folder")
    print("Download from: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
    sys.exit(1)

# Load YOLOv7 model with PyTorch 2.6+ compatibility
print("Loading YOLOv7 model...")
device = select_device('cpu')  # Use 'cpu' or '0' for GPU

# FIX: Load weights with weights_only=False for PyTorch 2.6+
try:
    # Method 1: Direct load with weights_only=False
    checkpoint = torch.load(str(WEIGHTS_PATH), map_location=device, weights_only=False)
    
    # Extract model from checkpoint
    if 'model' in checkpoint:
        model = checkpoint['model'].float().eval()
    else:
        model = checkpoint.float().eval()
    
    print("✓ Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTrying alternative loading method...")
    
    # Method 2: Use attempt_load with modified torch.load
    import torch.serialization
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    model = attempt_load(str(WEIGHTS_PATH), map_location=device)
    torch.load = original_load
    print("✓ Model loaded successfully with patched loader!")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

tracker = None

class SimpleKalmanTracker:
    def __init__(self, x, y, w, h):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        self.kf.statePost = np.array([
            [x + w // 2],
            [y + h // 2],
            [0],
            [0]
        ], np.float32)
        self.w, self.h = w, h
        
    def update(self, x, y, w, h):
        """Update tracker with new detection"""
        cx, cy = x + w // 2, y + h // 2
        self.kf.correct(np.array([[cx], [cy]], np.float32))
        self.w, self.h = w, h

    def predict(self):
        """Predict next position"""
        p = self.kf.predict()
        cx = int(p[0, 0])
        cy = int(p[1, 0])
        return cx - self.w // 2, cy - self.h // 2, self.w, self.h

frame_count = 0
print("Processing video... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    frame_count += 1
    img0 = frame.copy()
    
    # Preprocess image for YOLOv7
    img = letterbox(img0, 640, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # YOLOv7 Inference
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, CONFIDENCE, IOU_THRESHOLD)[0]

    # Process detections
    if pred is not None and len(pred):
        # Scale coordinates back to original image
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        
        # Find person (class 0)
        for *xyxy, conf, cls in pred:
            if int(cls) == 0:  # Person class
                x1, y1, x2, y2 = map(int, xyxy)
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                
                # Initialize or update tracker
                if tracker is None:
                    tracker = SimpleKalmanTracker(x, y, w, h)
                else:
                    tracker.update(x, y, w, h)
                
                # Draw detection box (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Detect: {conf:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

    # Draw tracking prediction (blue)
    if tracker:
        x, y, w, h = tracker.predict()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "YOLOv7 Kalman", (x, y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display frame info
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("YOLOv7 + Kalman Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✓ Processed {frame_count} frames")