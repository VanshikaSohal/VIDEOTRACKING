"""
3D UKF PERSON TRACKER - WITH DEPTH ESTIMATION
- 3D Unscented Kalman Filter (tracks X, Y, Z position)
- Estimates DEPTH using bounding box height changes
- Ground plane assumption for pedestrian tracking
- NO MACHINE LEARNING - Pure geometry + statistics
"""

import cv2
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# =======================
# CAMERA & PEDESTRIAN PARAMETERS (NO ML!)
# =======================
class CameraModel:
    """Camera calibration and 3D projection (pure geometry)"""
    def __init__(self, img_width, img_height, focal_length_px=None):
        self.img_width = img_width
        self.img_height = img_height
        
        # Estimate focal length if not provided (typical for consumer cameras)
        if focal_length_px is None:
            # Rule of thumb: focal_length ≈ image_width (for ~60° FOV)
            self.focal_length = img_width * 1.2
        else:
            self.focal_length = focal_length_px
        
        # Principal point (image center)
        self.cx = img_width / 2.0
        self.cy = img_height / 2.0
        
        print(f"📷 Camera Model:")
        print(f"   Focal length: {self.focal_length:.1f} px")
        print(f"   Principal point: ({self.cx:.1f}, {self.cy:.1f})")
    
    def estimate_depth_from_height(self, bbox_height_px, real_height_m=1.7):
        """
        Estimate depth (Z) from bounding box height using perspective projection
        
        Formula: depth = (focal_length × real_height) / bbox_height_pixels
        
        This is PURE GEOMETRY - no ML!
        """
        if bbox_height_px < 10:  # Avoid division by very small numbers
            return None
        
        depth = (self.focal_length * real_height_m) / bbox_height_px
        return depth
    
    def project_3d_to_2d(self, X, Y, Z):
        """
        Project 3D world point to 2D image point
        
        Standard pinhole camera model (pure geometry):
        u = fx * (X/Z) + cx
        v = fy * (Y/Z) + cy
        """
        if Z < 0.1:  # Avoid division by zero
            Z = 0.1
        
        u = self.focal_length * (X / Z) + self.cx
        v = self.focal_length * (Y / Z) + self.cy
        
        return u, v

class PedestrianPriors:
    """Statistical priors about pedestrians (NO ML - just statistics)"""
    def __init__(self):
        # Average adult height (meters) - statistical fact, not learned
        self.height_mean = 1.70  # meters
        self.height_std = 0.15   # standard deviation
        
        # Average adult width (shoulders, meters)
        self.width_mean = 0.50
        self.width_std = 0.10
        
        # Walking speed priors (m/s)
        self.walking_speed_mean = 1.4  # meters/second
        self.walking_speed_max = 2.5   # fast walk
        
        print(f"🚶 Pedestrian Priors (statistical, not learned):")
        print(f"   Height: {self.height_mean} ± {self.height_std} m")
        print(f"   Width: {self.width_mean} ± {self.width_std} m")
        print(f"   Walking speed: ~{self.walking_speed_mean} m/s")

def fx_3d(x, dt):
    F = np.array([
        [1, 0, 0, dt, 0,  0],   # X position
        [0, 1, 0, 0,  dt, 0],   # Y position  
        [0, 0, 1, 0,  0,  dt],  # Z position (DEPTH)
        [0, 0, 0, 1,  0,  0],   # X velocity
        [0, 0, 0, 0,  1,  0],   # Y velocity
        [0, 0, 0, 0,  0,  1]    # Z velocity (depth change)
    ], dtype=float)
    
    return F @ x

def hx_3d(x, camera):
    """
    3D Measurement function
    
    Converts 3D state [X, Y, Z, vX, vY, vZ] to 2D measurement [u, v, bbox_height]
    
    This uses perspective projection (pure geometry, no ML):
    u = fx * (X/Z) + cx
    v = fy * (Y/Z) + cy
    bbox_height = (focal_length * real_height) / Z
    """
    X, Y, Z = x[0], x[1], x[2]
    
    # Project 3D → 2D
    u, v = camera.project_3d_to_2d(X, Y, Z)
    
    # Predicted bounding box height from depth
    pedestrian_height = 1.7  # meters (statistical prior)
    predicted_bbox_height = (camera.focal_length * pedestrian_height) / Z
    
    return np.array([u, v, predicted_bbox_height])

class UKFTracker3D:
    """3D UKF Tracker with depth estimation (NO ML)"""
    def __init__(self, u, v, bbox_height, camera, pedestrian_priors, dt=1.0):
        self.camera = camera
        self.pedestrian_priors = pedestrian_priors
        self.dt = dt
        
        # 6D state: [X, Y, Z, vX, vY, vZ]
        self.dim_x = 6
        
        # 3D measurement: [u_pixel, v_pixel, bbox_height_pixels]
        self.dim_z = 3
        
        # Initialize depth estimate from bounding box height
        initial_depth = camera.estimate_depth_from_height(bbox_height, 
                                                          pedestrian_priors.height_mean)
        if initial_depth is None or initial_depth > 50 or initial_depth < 0.5:
            initial_depth = 5.0  # Default 5 meters
        
        # Estimate 3D position from 2D detection
        # Assume person's feet are at ground level (Y = 0 in world frame)
        X_init = (u - camera.cx) * initial_depth / camera.focal_length
        Y_init = 0.0  # Ground plane assumption
        Z_init = initial_depth
        
        print(f"   Initial 3D position: X={X_init:.2f}m, Y={Y_init:.2f}m, Z={Z_init:.2f}m")
        
        # Sigma points for UKF
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2., kappa=1)
        self.ukf = UKF(dim_x=self.dim_x, dim_z=self.dim_z,
                       fx=lambda x, dt: fx_3d(x, dt),
                       hx=lambda x: hx_3d(x, camera),
                       dt=dt, points=points)
        
        # Initial state: [X, Y, Z, vX, vY, vZ]
        self.ukf.x = np.array([X_init, Y_init, Z_init, 0, 0, 0], dtype=float)
        
        # Covariance matrices (tuned for 3D tracking)
        self.ukf.P *= 5.0  # Initial uncertainty
        
        # Measurement noise (pixels + bbox height)
        self.ukf.R = np.diag([10.0, 10.0, 20.0])  # [u_noise, v_noise, height_noise]
        
        # Process noise (3D motion uncertainty)
        self.ukf.Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])  # [X, Y, Z, vX, vY, vZ]
        
        # History for smoothing
        self.depth_history = [Z_init] * 3
    
    def predict(self):
        """Predict next 3D state"""
        self.ukf.predict()
        
        X, Y, Z = self.ukf.x[0], self.ukf.x[1], self.ukf.x[2]
        
        # Ensure depth stays positive
        if Z < 0.5:
            self.ukf.x[2] = 0.5
            Z = 0.5
        
        # Project 3D position back to 2D image
        u_pred, v_pred = self.camera.project_3d_to_2d(X, Y, Z)
        
        # Smooth depth estimate
        self.depth_history.append(Z)
        if len(self.depth_history) > 5:
            self.depth_history.pop(0)
        smooth_depth = np.median(self.depth_history)
        
        return int(u_pred), int(v_pred), smooth_depth
    
    def update(self, u, v, bbox_height):
        """Update with 2D measurement + bounding box height"""
        measurement = np.array([u, v, bbox_height], dtype=float)
        self.ukf.update(measurement)
        
        # Constrain depth to reasonable range
        if self.ukf.x[2] > 50:
            self.ukf.x[2] = 50
        elif self.ukf.x[2] < 0.5:
            self.ukf.x[2] = 0.5
    
    def get_3d_state(self):
        """Return current 3D position and velocity"""
        X, Y, Z = self.ukf.x[0], self.ukf.x[1], self.ukf.x[2]
        vX, vY, vZ = self.ukf.x[3], self.ukf.x[4], self.ukf.x[5]
        return X, Y, Z, vX, vY, vZ
    
    def get_predicted_bbox_size(self):
        """Calculate predicted bounding box size from 3D state"""
        Z = self.ukf.x[2]
        
        # Use perspective projection to get bbox size
        pred_height_px = (self.camera.focal_length * self.pedestrian_priors.height_mean) / Z
        pred_width_px = (self.camera.focal_length * self.pedestrian_priors.width_mean) / Z
        
        return int(pred_width_px), int(pred_height_px)

# =======================
# DETECTION FUNCTIONS (same as before)
# =======================
def detect_motion(frame, prev_frame):
    """Motion-based detection"""
    if prev_frame is None or frame.shape != prev_frame.shape:
        return []
    
    g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    g1 = cv2.GaussianBlur(g1, (21, 21), 0)
    g2 = cv2.GaussianBlur(g2, (21, 21), 0)
    
    diff = cv2.absdiff(g1, g2)
    _, th = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    th = cv2.dilate(th, kernel, iterations=5)
    th = cv2.erode(th, kernel, iterations=3)
    
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 400 and area < 50000:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = h / float(w) if w > 0 else 0
            
            if 1.1 < aspect_ratio < 5.5 and h > 18 and w > 12:
                padding_w = int(w * 0.6)
                padding_h = int(h * 0.4)
                x = max(0, x - padding_w)
                y = max(0, y - padding_h)
                w = w + 2 * padding_w
                h = h + 2 * padding_h
                boxes.append((x, y, w, h))
    
    return boxes

def detect_bg(frame, bg):
    """Background subtraction"""
    fg = bg.apply(frame, learningRate=0.012)
    
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
            
            if 1.1 < aspect_ratio < 5.5 and h > 22 and w > 12:
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
print("3D UKF PERSON TRACKER - WITH DEPTH ESTIMATION")
print("="*70)
print("\n🎯 Features:")
print("   • 3D Unscented Kalman Filter (tracks X, Y, Z)")
print("   • Depth estimation from bounding box height changes")
print("   • Ground plane assumption for pedestrians")
print("   • Handles receding/approaching motion")
print("   • NO MACHINE LEARNING - Pure geometry + statistics")
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

# Initialize camera model and pedestrian priors (NO ML!)
camera = CameraModel(W, H)
pedestrian_priors = PedestrianPriors()
print()

bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=18, detectShadows=False)

tracker = None
tracking = False
lost_frames = 0
consecutive_detections = 0
min_confidence = 3

total_frames = 0
frames_with_detection = 0
accurate_detections = 0

bbox_history = []
max_bbox_history = 4

print("🎬 Starting 3D tracking...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    total_frames += 1
    used_measurement = False
    detection_bbox = None
    
    if not tracking:
        motion_dets = detect_motion(frame, prev_frame)
        bg_dets = detect_bg(frame, bg)
        
        all_dets = motion_dets + bg_dets
        all_dets = combine_detections(all_dets, max_dist=200)
        all_dets = [b for b in all_dets if b[2] > 20 and b[3] > 45]
        
        if total_frames % 10 == 0:
            print(f"Frame {total_frames}: Searching... detections={len(all_dets)}")
        
        if all_dets:
            x, y, w, h = max(all_dets, key=lambda b: b[2]*b[3])
            cx, cy = x + w//2, y + h//2
            
            consecutive_detections += 1
            
            if consecutive_detections >= min_confidence:
                tracker = UKFTracker3D(cx, cy, h, camera, pedestrian_priors, 
                                      dt=1.0/fps if fps > 0 else 0.033)
                tracking = True
                lost_frames = 0
                bbox_history = [(x, y, w, h)]
                
                print(f"✅ Frame {total_frames}: Person DETECTED - 3D Tracking STARTED")
        else:
            consecutive_detections = 0
    
    else:
        pred_u, pred_v, pred_depth = tracker.predict()
        pred_u = max(0, min(pred_u, W - 1))
        pred_v = max(0, min(pred_v, H - 1))
        
        pred_bbox_w, pred_bbox_h = tracker.get_predicted_bbox_size()
        
        ROI = min(280 + lost_frames * 25, 650)
        x1, y1 = max(0, pred_u - ROI), max(0, pred_v - ROI)
        x2, y2 = min(W, pred_u + ROI), min(H, pred_v + ROI)
        
        roi = frame[y1:y2, x1:x2]
        prev_roi = prev_frame[y1:y2, x1:x2] if prev_frame.shape == frame.shape else roi
        
        motion_dets = detect_motion(roi, prev_roi)
        bg_dets = detect_bg(roi, bg)
        
        dets = motion_dets + bg_dets
        dets = combine_detections(dets, max_dist=200)
        
        best_dist = 1e9
        best_center = None
        best_box = None
        
        for (x, y, w, h) in dets:
            cu = x1 + x + w//2
            cv = y1 + y + h//2
            d = np.hypot(cu - pred_u, cv - pred_v)
            
            if d < best_dist:
                best_dist = d
                best_center = (cu, cv, h)
                best_box = (x1 + x, y1 + y, w, h)
        
        dist_threshold = 0.65 * ROI
        
        if best_center and best_dist < dist_threshold:
            cu, cv, bbox_h = best_center
            tracker.update(cu, cv, bbox_h)
            
            used_measurement = True
            detection_bbox = best_box
            frames_with_detection += 1
            
            bbox_history.append(best_box)
            if len(bbox_history) > max_bbox_history:
                bbox_history.pop(0)
            
            if best_dist < 45:
                accurate_detections += 1
            
            lost_frames = 0
        else:
            lost_frames += 1
        
        X, Y, Z, vX, vY, vZ = tracker.get_3d_state()
        velocity_3d = np.sqrt(vX**2 + vY**2 + vZ**2)
        
        if bbox_history and len(bbox_history) >= 2:
            smooth_x = int(np.median([b[0] for b in bbox_history]))
            smooth_y = int(np.median([b[1] for b in bbox_history]))
            smooth_w = int(np.median([b[2] for b in bbox_history]))
            smooth_h = int(np.median([b[3] for b in bbox_history]))
            detection_bbox = (smooth_x, smooth_y, smooth_w, smooth_h)
        
        bw, bh = pred_bbox_w, pred_bbox_h
        cv2.rectangle(frame,
                      (pred_u - bw//2, pred_v - bh//2),
                      (pred_u + bw//2, pred_v + bh//2),
                      (0, 0, 255), 3)
        cv2.putText(frame, f"PRED (Z={Z:.1f}m)", 
                    (pred_u - bw//2, pred_v - bh//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if used_measurement and detection_bbox:
            dx, dy, dw, dh = detection_bbox
            cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 3)
            cv2.putText(frame, f"DETECT (h={dh}px)", (dx, dy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if lost_frames > 50:
            tracking = False
            bbox_history.clear()
            consecutive_detections = 0
            print(f"\n⚠️ Frame {total_frames}: Tracking LOST - Restarting")
        elif lost_frames > 0 and lost_frames % 20 == 0:
            print(f"⚠️ Frame {total_frames}: No detection for {lost_frames} frames")
        
        if total_frames % 10 == 0:
            print(f"Frame {total_frames}: 2D=({pred_u},{pred_v}) 3D=({X:.1f},{Y:.1f},{Z:.1f}m) "
                  f"v={velocity_3d:.2f}m/s det={used_measurement}")
    
    panel_h = 160
    cv2.rectangle(frame, (0, 0), (W, panel_h), (40, 40, 40), -1)
    
    cv2.putText(frame, f"Frame: {total_frames} | Detections: {frames_with_detection}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    status = "TRACKING (3D)" if tracking else "SEARCHING"
    status_color = (0, 255, 0) if used_measurement else (0, 165, 255)
    cv2.putText(frame, f"Status: {status}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    if tracking:
        X, Y, Z, vX, vY, vZ = tracker.get_3d_state()
        cv2.putText(frame, f"3D Pos: X={X:.1f}m Y={Y:.1f}m Z={Z:.1f}m",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
        
        velocity_3d = np.sqrt(vX**2 + vY**2 + vZ**2)
        cv2.putText(frame, f"Velocity: {velocity_3d:.2f} m/s | vZ={vZ:.2f} m/s",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
    
    cv2.imshow("3D UKF Person Tracker", frame)
    
    prev_frame = frame.copy()
    
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
    elif key == ord(' '):
        cv2.waitKey(0)
    elif key == ord('r'):
        tracking = False
        tracker = None
        lost_frames = 0
        bbox_history.clear()
        consecutive_detections = 0
        bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=18, detectShadows=False)
        print("\n🔄 RESET\n")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("FINAL STATISTICS - 3D TRACKING")
print("="*70)
print(f"Total Frames: {total_frames}")
print(f"Frames with Detection: {frames_with_detection}")
print(f"Detection Rate: {(frames_with_detection/total_frames)*100:.1f}%")
if frames_with_detection > 0:
    print(f"Detection Accuracy: {(accurate_detections/frames_with_detection)*100:.1f}%")
    print(f"Accurate Detections: {accurate_detections}/{frames_with_detection}")
print("="*70)
print("\n💡 This uses NO MACHINE LEARNING - only:")
print("   • Perspective projection geometry")
print("   • Statistical priors (pedestrian height ~1.7m)")
print("   • Unscented Kalman Filter (pure math)")
print("="*70)