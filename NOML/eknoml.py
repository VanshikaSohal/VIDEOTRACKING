"""
EKF PERSON TRACKER - WITHOUT HOG
- Extended Kalman Filter for non-linear tracking
- Motion Detection + Background Subtraction only
- Adaptive ROI and dynamic bounding boxes
"""

import cv2
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF

# =======================
# EKF TRACKER CLASS
# =======================
def fx(x, dt):
    """State transition function for constant velocity model"""
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return F @ x

def hx(x):
    """Measurement function: return position only"""
    return np.array([x[0], x[1]])

def H_jacobian(x):
    """Jacobian of measurement function"""
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return H

class EKFTracker:
    def __init__(self, x, y, dt=1.0):
        self.dt = dt
        self.ekf = EKF(dim_x=4, dim_z=2)
        
        # Initial state [x, y, vx, vy]
        self.ekf.x = np.array([x, y, 0, 0], dtype=float)
        
        # State transition matrix
        self.ekf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # Measurement matrix
        self.ekf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Uncertainties - TUNED FOR BETTER TRACKING
        self.ekf.P *= 10.0                # Initial state covariance
        self.ekf.R = np.eye(2) * 5.0      # Measurement noise
        self.ekf.Q = np.eye(4) * 0.05     # Process noise
        
        # Velocity smoothing
        self.prev_positions = [(x, y)] * 3
    
    def predict(self):
        self.ekf.predict()
        pred_x = int(self.ekf.x[0])
        pred_y = int(self.ekf.x[1])
        
        # Smooth prediction using recent history
        self.prev_positions.append((pred_x, pred_y))
        if len(self.prev_positions) > 3:
            self.prev_positions.pop(0)
        
        smooth_x = int(np.mean([p[0] for p in self.prev_positions]))
        smooth_y = int(np.mean([p[1] for p in self.prev_positions]))
        
        return smooth_x, smooth_y
    
    def update(self, x, y):
        z = np.array([x, y], dtype=float)
        self.ekf.update(z, HJacobian=H_jacobian, Hx=hx)
    
    def get_velocity(self):
        """Return current velocity estimate"""
        return self.ekf.x[2], self.ekf.x[3]

# =======================
# DETECTION FUNCTIONS
# =======================
def detect_motion(frame, prev_frame):
    """Motion-based detection with improved morphology"""
    if prev_frame is None or frame.shape != prev_frame.shape:
        return []
    
    g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    g1 = cv2.GaussianBlur(g1, (21, 21), 0)
    g2 = cv2.GaussianBlur(g2, (21, 21), 0)
    
    diff = cv2.absdiff(g1, g2)
    
    # Lower threshold for subtle motion
    _, th = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    
    # Stronger morphology to connect body parts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    th = cv2.dilate(th, kernel, iterations=5)
    th = cv2.erode(th, kernel, iterations=3)
    
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 400 and area < 50000:  # Filter by area
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = h / float(w) if w > 0 else 0
            
            # Filter by aspect ratio (person-like shapes)
            if 1.1 < aspect_ratio < 5.5 and h > 18 and w > 12:
                # Expand bbox generously
                padding_w = int(w * 0.6)
                padding_h = int(h * 0.4)
                x = max(0, x - padding_w)
                y = max(0, y - padding_h)
                w = w + 2 * padding_w
                h = h + 2 * padding_h
                boxes.append((x, y, w, h))
    
    return boxes

def detect_bg(frame, bg):
    """Background subtraction with improved filtering"""
    # Higher learning rate for dynamic backgrounds
    fg = bg.apply(frame, learningRate=0.012)
    
    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 500 and area < 45000:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = h / float(w) if w > 0 else 0
            
            # Person-like aspect ratio
            if 1.1 < aspect_ratio < 5.5 and h > 22 and w > 12:
                # Expand bbox
                padding_w = int(w * 0.6)
                padding_h = int(h * 0.4)
                x = max(0, x - padding_w)
                y = max(0, y - padding_h)
                w = w + 2 * padding_w
                h = h + 2 * padding_h
                boxes.append((x, y, w, h))
    
    return boxes

def combine_detections(boxes, max_dist=180):
    """Merge nearby detections"""
    if not boxes:
        return []
    
    merged = []
    used = set()
    
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if i in used:
            continue
        
        similar = [(x1, y1, w1, h1)]
        cx1, cy1 = x1 + w1//2, y1 + h1//2
        
        for j, (x2, y2, w2, h2) in enumerate(boxes[i+1:], i+1):
            if j in used:
                continue
            
            cx2, cy2 = x2 + w2//2, y2 + h2//2
            dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            
            if dist < max_dist:
                similar.append((x2, y2, w2, h2))
                used.add(j)
        
        if similar:
            # Create encompassing box
            all_x = [b[0] for b in similar]
            all_y = [b[1] for b in similar]
            all_x2 = [b[0] + b[2] for b in similar]
            all_y2 = [b[1] + b[3] for b in similar]
            
            merged_x = min(all_x)
            merged_y = min(all_y)
            merged_w = max(all_x2) - merged_x
            merged_h = max(all_y2) - merged_y
            
            merged.append((merged_x, merged_y, merged_w, merged_h))
    
    return merged

# =======================
# MAIN TRACKING LOOP
# =======================
print("="*70)
print("EKF PERSON TRACKER - NO HOG VERSION")
print("="*70)
print("\n🎯 Features:")
print("   • Extended Kalman Filter for non-linear tracking")
print("   • Motion Detection + Background Subtraction")
print("   • Adaptive ROI that grows with uncertainty")
print("   • Dynamic bounding box sizing")
print("\n🎮 Controls: ESC=Quit | SPACE=Pause | R=Reset")
print("="*70)

VIDEO_FILE = "realcurvedvideo.mp4"

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"\n❌ Cannot open '{VIDEO_FILE}'")
    exit()

ret, frame = cap.read()
if not ret:
    print("❌ Cannot read first frame")
    exit()

prev_frame = frame.copy()
H, W = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"\n📹 Video: {W}x{H} @ {fps:.1f} FPS\n")

# Background subtractor with optimized parameters
bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=18, detectShadows=False)

tracker = None
tracking = False
lost_frames = 0
consecutive_detections = 0
min_confidence = 3  # Need 3 consecutive detections to start tracking

# Statistics
total_frames = 0
frames_with_detection = 0
accurate_detections = 0

# Detection history for bbox smoothing
bbox_history = []
max_bbox_history = 4

print("🎬 Starting tracking...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    total_frames += 1
    used_measurement = False
    detection_bbox = None
    
    # PHASE 1: INITIAL DETECTION (searching for person)
    if not tracking:
        # Run detectors on full frame
        motion_dets = detect_motion(frame, prev_frame)
        bg_dets = detect_bg(frame, bg)
        
        all_dets = motion_dets + bg_dets
        all_dets = combine_detections(all_dets, max_dist=200)
        
        # Filter by size
        all_dets = [b for b in all_dets if b[2] > 20 and b[3] > 45]
        
        if total_frames % 10 == 0:
            print(f"Frame {total_frames}: Searching... detections={len(all_dets)}")
        
        if all_dets:
            # Take largest detection
            x, y, w, h = max(all_dets, key=lambda b: b[2]*b[3])
            cx, cy = x + w//2, y + h//2
            
            consecutive_detections += 1
            
            if consecutive_detections >= min_confidence:
                tracker = EKFTracker(cx, cy, dt=1.0/fps if fps > 0 else 0.033)
                tracking = True
                lost_frames = 0
                bbox_history = [(x, y, w, h)]
                
                print(f"✅ Frame {total_frames}: Person DETECTED at ({cx}, {cy}) - Tracking STARTED")
        else:
            consecutive_detections = 0
        
        if total_frames % 30 == 0 and not tracking:
            print(f"Frame {total_frames}: Still searching...")
    
    # PHASE 2: TRACKING MODE (person locked)
    else:
        pred_x, pred_y = tracker.predict()
        pred_x = max(0, min(pred_x, W - 1))
        pred_y = max(0, min(pred_y, H - 1))
        
        # Adaptive ROI that grows with uncertainty
        ROI = min(280 + lost_frames * 25, 650)
        x1, y1 = max(0, pred_x - ROI), max(0, pred_y - ROI)
        x2, y2 = min(W, pred_x + ROI), min(H, pred_y + ROI)
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        prev_roi = prev_frame[y1:y2, x1:x2] if prev_frame.shape == frame.shape else roi
        
        # Run detectors on ROI
        motion_dets = detect_motion(roi, prev_roi)
        bg_dets = detect_bg(roi, bg)
        
        dets = motion_dets + bg_dets
        dets = combine_detections(dets, max_dist=200)
        
        # Find best detection (closest to prediction)
        best_dist = 1e9
        best_center = None
        best_box = None
        
        for (x, y, w, h) in dets:
            # Convert ROI coordinates to frame coordinates
            cx = x1 + x + w//2
            cy = y1 + y + h//2
            d = np.hypot(cx - pred_x, cy - pred_y)
            
            if d < best_dist:
                best_dist = d
                best_center = (cx, cy)
                best_box = (x1 + x, y1 + y, w, h)
        
        # Adaptive distance threshold (more lenient when lost)
        dist_threshold = 0.65 * ROI
        
        if best_center and best_dist < dist_threshold:
            # Good detection found - update tracker
            tracker.update(best_center[0], best_center[1])
            used_measurement = True
            detection_bbox = best_box
            frames_with_detection += 1
            
            # Smooth bbox
            bbox_history.append(best_box)
            if len(bbox_history) > max_bbox_history:
                bbox_history.pop(0)
            
            if best_dist < 45:
                accurate_detections += 1
            
            lost_frames = 0
        else:
            lost_frames += 1
        
        # Get velocity for adaptive bbox sizing
        vx, vy = tracker.get_velocity()
        velocity_mag = np.sqrt(vx**2 + vy**2)
        
        # Dynamic prediction box sizing based on distance from camera
        scale = pred_y / H  # Objects farther = smaller
        base_w = 65
        base_h = 130
        bw = int(base_w + scale * 110 + velocity_mag * 5)
        bh = int(base_h + scale * 200 + velocity_mag * 8)
        
        bw = max(50, min(bw, 200))
        bh = max(100, min(bh, 350))
        
        # Smooth detection box
        if bbox_history and len(bbox_history) >= 2:
            smooth_x = int(np.median([b[0] for b in bbox_history]))
            smooth_y = int(np.median([b[1] for b in bbox_history]))
            smooth_w = int(np.median([b[2] for b in bbox_history]))
            smooth_h = int(np.median([b[3] for b in bbox_history]))
            detection_bbox = (smooth_x, smooth_y, smooth_w, smooth_h)
        
        # DRAW PREDICTION BOX (RED)
        cv2.rectangle(frame,
                      (pred_x - bw//2, pred_y - bh//2),
                      (pred_x + bw//2, pred_y + bh//2),
                      (0, 0, 255), 3)
        cv2.putText(frame, "PRED", (pred_x - bw//2, pred_y - bh//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # DRAW DETECTION BOX (GREEN)
        if used_measurement and detection_bbox:
            dx, dy, dw, dh = detection_bbox
            cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 3)
            cv2.putText(frame, "DETECT", (dx, dy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Reset if lost for too long
        if lost_frames > 50:
            tracking = False
            bbox_history.clear()
            consecutive_detections = 0
            print(f"\n⚠️ Frame {total_frames}: Tracking LOST after {lost_frames} frames - Restarting")
        elif lost_frames > 0 and lost_frames % 20 == 0:
            print(f"⚠️ Frame {total_frames}: No detection for {lost_frames} frames (predicting...)")
        
        if total_frames % 10 == 0:
            print(f"Frame {total_frames}: pred=({pred_x},{pred_y}) det={used_measurement} lost={lost_frames}")
    
    # VISUALIZATION PANEL
    panel_h = 120
    cv2.rectangle(frame, (0, 0), (W, panel_h), (40, 40, 40), -1)
    
    cv2.putText(frame, f"Frame: {total_frames} | Detections: {frames_with_detection}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    status = "TRACKING" if tracking else "SEARCHING"
    status_color = (0, 255, 0) if used_measurement else (0, 165, 255)
    cv2.putText(frame, f"Status: {status}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    if tracking and lost_frames > 0:
        cv2.putText(frame, f"Lost: {lost_frames} frames",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Show frame
    cv2.imshow("EKF Person Tracker - No HOG", frame)
    
    prev_frame = frame.copy()
    
    # Keyboard controls
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE - pause
        cv2.waitKey(0)
    elif key == ord('r'):  # R - reset
        tracking = False
        tracker = None
        lost_frames = 0
        bbox_history.clear()
        consecutive_detections = 0
        bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=18, detectShadows=False)
        print("\n🔄 RESET\n")

cap.release()
cv2.destroyAllWindows()

# FINAL STATISTICS
print("\n" + "="*70)
print("FINAL STATISTICS")
print("="*70)
print(f"Total Frames: {total_frames}")
print(f"Frames with Detection: {frames_with_detection}")
print(f"Detection Rate: {(frames_with_detection/total_frames)*100:.1f}%")
if frames_with_detection > 0:
    print(f"Detection Accuracy: {(accurate_detections/frames_with_detection)*100:.1f}%")
    print(f"Accurate Detections: {accurate_detections}/{frames_with_detection}")
print("="*70)