"""
SIMPLIFIED YOLO + KALMAN TRACKER
Perfect for beginners - heavily commented version
"""

import cv2
import numpy as np
from ultralytics import YOLO

# =======================
# SIMPLE KALMAN TRACKER
# =======================
class SimpleKalmanTracker:
    """Simple Kalman filter for tracking a single object"""
    
    def __init__(self, x, y, w, h):
        # Create Kalman filter with 4 states and 2 measurements
        # States: [x, y, dx, dy] - position and velocity
        # Measurements: [x, y] - observed position
        self.kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
        
        # How state changes (position += velocity)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x_new = x_old + dx
            [0, 1, 0, 1],  # y_new = y_old + dy
            [0, 0, 1, 0],  # dx_new = dx_old
            [0, 0, 0, 1]   # dy_new = dy_old
        ], np.float32)
        
        # What we can measure (just x and y position)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 1, 0, 0]   # measure y
        ], np.float32)
        
        # How much we trust predictions vs measurements
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Initial position (center of bounding box)
        cx = x + w // 2
        cy = y + h // 2
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        
        # Store box dimensions
        self.w = w
        self.h = h
        
    def predict(self):
       """Predict where object will be in next frame"""
       pred = self.kf.predict()        # shape = (4,1)
       pred = pred.flatten()           # convert to 1D array: [x, y, dx, dy]
    
       cx = int(pred[0])
       cy = int(pred[1])
    
       # Convert center to top-left corner
       x = cx - self.w // 2
       y = cy - self.h // 2
    
       return x, y, self.w, self.h

    
    def update(self, x, y, w, h):
        """Update filter with new detection"""
        # Update box size
        self.w = w
        self.h = h
        
        # Convert to center
        cx = x + w // 2
        cy = y + h // 2
        
        # Update Kalman filter
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(measurement)


# =======================
# MAIN PROGRAM
# =======================
def main():
    # =============== CONFIGURATION ===============
    VIDEO_PATH = "realvideo.mp4"  # Your video file (or 0 for webcam)
    YOLO_MODEL = "yolov8n.pt"     # YOLO model (n=fastest, x=most accurate)
    CONFIDENCE = 0.5              # Detection confidence (0.0 to 1.0)
    TRACK_CLASS = 0               # What to track: 0=person, 2=car, etc.
    # ============================================
    
    print("🚀 Starting YOLO + Kalman Tracker...")
    print(f"📹 Video: {VIDEO_PATH}")
    print(f"🎯 Tracking class: {TRACK_CLASS}")
    print(f"⚙️  Confidence threshold: {CONFIDENCE}")
    
    # Load YOLO model
    print("\n📦 Loading YOLO model...")
    model = YOLO(YOLO_MODEL)
    print("✅ Model loaded!")
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video '{VIDEO_PATH}'")
        print("💡 Make sure the video file exists or use 0 for webcam")
        return
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📊 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print("\n▶️  Press 'q' or ESC to quit\n")
    
    # Tracking variables
    tracker = None  # Kalman tracker (created when object detected)
    tracking = False  # Are we currently tracking?
    lost_frames = 0  # Frames since last detection
    MAX_LOST = 30  # Stop tracking after this many frames without detection
    
    frame_number = 0
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n🏁 End of video")
            break
        
        frame_number += 1
        
        # ===== DETECTION PHASE =====
        if not tracking:
            # Not tracking yet - look for object
            results = model(frame, conf=CONFIDENCE, verbose=False)
            
            # Check all detections
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    
                    # Is this the class we want to track?
                    if class_id == TRACK_CLASS:
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x = int(x1)
                        y = int(y1)
                        w = int(x2 - x1)
                        h = int(y2 - y1)
                        
                        # Create tracker
                        tracker = SimpleKalmanTracker(x, y, w, h)
                        tracking = True
                        lost_frames = 0
                        print(f"✅ Object detected at frame {frame_number} - Starting tracking!")
                        break
                
                if tracking:
                    break
        
        # ===== TRACKING PHASE =====
        else:
            # Get prediction from Kalman filter
            pred_x, pred_y, pred_w, pred_h = tracker.predict()
            
            # Look for object near prediction
            # Create search region (ROI = Region of Interest)
            search_size = 150  # Pixels to search around prediction
            roi_x1 = max(0, pred_x - search_size)
            roi_y1 = max(0, pred_y - search_size)
            roi_x2 = min(width, pred_x + pred_w + search_size)
            roi_y2 = min(height, pred_y + pred_h + search_size)
            
            # Detect in search region
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            results = model(roi, conf=CONFIDENCE, verbose=False)
            
            # Find closest detection to prediction
            best_detection = None
            best_distance = float('inf')
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    
                    if class_id == TRACK_CLASS:
                        # Get bounding box (relative to ROI)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Convert to full frame coordinates
                        x = int(x1) + roi_x1
                        y = int(y1) + roi_y1
                        w = int(x2 - x1)
                        h = int(y2 - y1)
                        
                        # Calculate distance to prediction
                        det_cx = x + w // 2
                        det_cy = y + h // 2
                        pred_cx = pred_x + pred_w // 2
                        pred_cy = pred_y + pred_h // 2
                        
                        distance = np.sqrt((det_cx - pred_cx)**2 + (det_cy - pred_cy)**2)
                        
                        # Is this the closest detection?
                        if distance < best_distance:
                            best_distance = distance
                            best_detection = (x, y, w, h)
            
            # Did we find a detection?
            if best_detection and best_distance < 200:
                # Update tracker with detection
                x, y, w, h = best_detection
                tracker.update(x, y, w, h)
                lost_frames = 0
                
                # Draw GREEN box (detected + tracked)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, "TRACKING (detected)", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # No detection - use prediction only
                lost_frames += 1
                
                # Draw YELLOW box (prediction only)
                cv2.rectangle(frame, (pred_x, pred_y), 
                            (pred_x + pred_w, pred_y + pred_h), (0, 255, 255), 3)
                cv2.putText(frame, f"PREDICTING ({lost_frames} frames)", 
                           (pred_x, pred_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Lost track?
                if lost_frames > MAX_LOST:
                    tracking = False
                    tracker = None
                    print(f"❌ Lost track at frame {frame_number}")
        
        # ===== DISPLAY INFO =====
        status = "TRACKING" if tracking else "SEARCHING"
        color = (0, 255, 0) if tracking else (0, 0, 255)
        
        cv2.putText(frame, f"Frame: {frame_number} | Status: {status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if tracking:
            cv2.putText(frame, f"Lost frames: {lost_frames}/{MAX_LOST}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("YOLO + Kalman Tracking", frame)
        
        # Check for quit (q or ESC)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n⏹️  Stopped by user")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Processed {frame_number} frames")
    print("👋 Done!")


if __name__ == "__main__":
    main()