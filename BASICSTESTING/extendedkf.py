import cv2
import numpy as np

# Detect White Objects Function
def detect_white_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask

def detect_white_car_candidates(frame, object_specs):
    """Detect ALL white car candidates and return them sorted by quality"""
    mask = detect_white_objects(frame)
    
    if object_specs.get('region_of_interest') is not None:
        x1, y1, x2, y2 = object_specs['region_of_interest']
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        mask = roi_mask
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        min_area = object_specs['min_area'] * 0.4
        max_area = object_specs.get('max_area', float('inf'))
        
        if area < min_area or area > max_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        if h > 0:
            aspect_ratio = w / h
            min_ar, max_ar = object_specs['aspect_ratio']
            expanded_min_ar = min_ar * 0.5
            expanded_max_ar = max_ar * 1.5
            
            if expanded_min_ar <= aspect_ratio <= expanded_max_ar:
                cx = x + w // 2
                cy = y + h // 2
                quality_score = area
                
                valid_candidates.append({
                    'contour': cnt,
                    'center': (cx, cy),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'quality_score': quality_score
                })
    
    valid_candidates.sort(key=lambda d: d['quality_score'], reverse=True)
    return valid_candidates

# ============================================
# EXTENDED KALMAN FILTER IMPLEMENTATION
# ============================================

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for non-linear motion tracking
    
    State vector: [x, y, vx, vy, ax, ay]
    - x, y: position
    - vx, vy: velocity
    - ax, ay: acceleration (handles curves and turns)
    
    Non-linear motion model:
    - x_new = x + vx*dt + 0.5*ax*dt²
    - y_new = y + vy*dt + 0.5*ay*dt²
    - vx_new = vx + ax*dt
    - vy_new = vy + ay*dt
    - ax_new = ax (with some decay)
    - ay_new = ay (with some decay)
    """
    
    def __init__(self, x, y, dt=1.0):
        # State: [x, y, vx, vy, ax, ay]
        self.state = np.array([[x], [y], [0], [0], [0], [0]], dtype=np.float32)
        
        # Time step
        self.dt = dt
        
        # State covariance matrix (uncertainty)
        self.P = np.eye(6, dtype=np.float32) * 1.0
        
        # Process noise covariance
        self.Q = np.eye(6, dtype=np.float32)
        self.Q[0, 0] = 0.05  # x position noise
        self.Q[1, 1] = 0.05  # y position noise
        self.Q[2, 2] = 0.1   # x velocity noise
        self.Q[3, 3] = 0.1   # y velocity noise
        self.Q[4, 4] = 0.5   # x acceleration noise
        self.Q[5, 5] = 0.5   # y acceleration noise
        
        # Measurement noise covariance (how much we trust measurements)
        self.R = np.eye(2, dtype=np.float32) * 1.0
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
    
    def predict(self):
        """
        EKF Prediction Step with non-linear motion model
        """
        x, y, vx, vy, ax, ay = self.state.flatten()
        dt = self.dt
        
        # Non-linear state transition
        x_new = x + vx * dt + 0.5 * ax * dt**2
        y_new = y + vy * dt + 0.5 * ay * dt**2
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        ax_new = ax * 0.9  # Acceleration decay
        ay_new = ay * 0.9
        
        # Update state
        self.state = np.array([[x_new], [y_new], [vx_new], [vy_new], [ax_new], [ay_new]], dtype=np.float32)
        
        # Jacobian of state transition (linearization)
        F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 0.9, 0],
            [0, 0, 0, 0, 0, 0.9]
        ], dtype=np.float32)
        
        # Update covariance: P = F*P*F' + Q
        self.P = F @ self.P @ F.T + self.Q
        
        return int(self.state[0, 0]), int(self.state[1, 0])
    
    def update(self, measurement_x, measurement_y):
        """
        EKF Update Step
        """
        # Measurement vector
        z = np.array([[measurement_x], [measurement_y]], dtype=np.float32)
        
        # Predicted measurement
        z_pred = self.H @ self.state
        
        # Innovation (measurement residual)
        y = z - z_pred
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
    
    def get_state(self):
        """Get current state"""
        return {
            'x': float(self.state[0, 0]),
            'y': float(self.state[1, 0]),
            'vx': float(self.state[2, 0]),
            'vy': float(self.state[3, 0]),
            'ax': float(self.state[4, 0]),
            'ay': float(self.state[5, 0])
        }

# ============================================
# MAIN PROGRAM
# ============================================

print("=" * 70)
print("EXTENDED KALMAN FILTER (EKF) - WHITE CAR TRACKING")
print("=" * 70)
print("\n🎯 EKF FEATURES:")
print("  ✓ Handles non-linear motion (curves, turns)")
print("  ✓ Tracks acceleration (not just velocity)")
print("  ✓ Better prediction for maneuvering vehicles")
print("  ✓ State: [x, y, vx, vy, ax, ay]")
print("\nCONTROLS:")
print("  Q/ESC - Quit | SPACE - Pause | D - Mask | R - Reset")
print("=" * 70)

object_specs = {
    'color': 'white',
    'min_area': 3000,
    'max_area': 50000,
    'aspect_ratio': (0.3, 4.0),
    'region_of_interest': None
}

cap = cv2.VideoCapture("challenge.mp4")
if not cap.isOpened():
    print("\n❌ ERROR: Cannot open video file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n📹 VIDEO: {frame_width}x{frame_height} @ {fps} FPS\n")

# Tracking variables
frame_count = 0
ekf = None
tracking_initialized = False
tracking_active = False
lost_frames = 0
max_lost_frames = 45
recovery_threshold = 15
recovery_mode = False

# Statistics
total_detections = 0
total_predictions = 0
recovery_attempts = 0
successful_recoveries = 0

show_mask = False

# Trajectory history for visualization
trajectory = []
max_trajectory_length = 100

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"\n🏁 Video ended at frame {frame_count}")
        break
    
    frame_count += 1
    
    detection_found = False
    detected_x, detected_y = None, None
    pred_x, pred_y = None, None
    detection_info = None
    candidates = []
    search_x1, search_y1, search_x2, search_y2 = 0, 0, 0, 0
    detections_this_frame = 0
    
    # PHASE 1: INITIAL DETECTION
    if not tracking_initialized:
        candidates = detect_white_car_candidates(frame, object_specs)
        
        if len(candidates) > 0:
            detection_info = candidates[0]
            detection_found = True
            detected_x, detected_y = detection_info['center']
            
            if detections_this_frame == 0:
                total_detections += 1
                detections_this_frame += 1
            
            # Initialize EKF
            ekf = ExtendedKalmanFilter(detected_x, detected_y, dt=1.0)
            tracking_initialized = True
            tracking_active = True
            
            print(f"\n✓ CAR DETECTED at frame {frame_count}")
            print(f"  Position: ({detected_x}, {detected_y})")
            print(f"  EKF initialized with 6-state model [x, y, vx, vy, ax, ay]\n")
        else:
            detection_found = False
    
    # PHASE 2: TRACKING MODE
    else:
        # EKF PREDICTION
        pred_x, pred_y = ekf.predict()
        total_predictions += 1
        
        pred_x = max(0, min(pred_x, frame.shape[1] - 1))
        pred_y = max(0, min(pred_y, frame.shape[0] - 1))
        
        # Get current state for adaptive search
        state = ekf.get_state()
        vx, vy = state['vx'], state['vy']
        ax, ay = state['ax'], state['ay']
        speed = np.sqrt(vx**2 + vy**2)
        accel = np.sqrt(ax**2 + ay**2)
        
        # RECOVERY MODE
        if lost_frames >= recovery_threshold:
            recovery_mode = True
            recovery_attempts += 1
            search_x1, search_y1 = 0, 0
            search_x2, search_y2 = frame.shape[1], frame.shape[0]
            
            if lost_frames == recovery_threshold:
                print(f"\n🔍 Frame {frame_count}: RECOVERY MODE")
        else:
            recovery_mode = False
            
            # Adaptive search based on speed and acceleration
            base_search_size = max(200, int(speed * 15), int(accel * 30))
            search_expansion = lost_frames * 25
            search_size = min(base_search_size + search_expansion, 600)
            
            search_x1 = max(0, pred_x - search_size)
            search_y1 = max(0, pred_y - search_size)
            search_x2 = min(frame.shape[1], pred_x + search_size)
            search_y2 = min(frame.shape[0], pred_y + search_size)
        
        search_specs = object_specs.copy()
        search_specs['region_of_interest'] = (search_x1, search_y1, search_x2, search_y2)
        
        candidates = detect_white_car_candidates(frame, search_specs)
        
        # Find best match
        best_candidate = None
        best_score = -float('inf')
        
        for candidate in candidates:
            cx, cy = candidate['center']
            distance = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
            
            max_distance = 200 + (lost_frames * 15)
            if distance > max_distance and not recovery_mode:
                continue
            
            distance_score = 1.0 - (distance / max_distance) if max_distance > 0 else 0
            
            if distance_score > best_score:
                best_score = distance_score
                best_candidate = candidate
        
        min_score_threshold = 0.3 if recovery_mode else 0.5
        
        if best_candidate is not None and best_score > min_score_threshold:
            detection_info = best_candidate
            detection_found = True
            detected_x, detected_y = detection_info['center']
            
            if detections_this_frame == 0:
                total_detections += 1
                detections_this_frame += 1
            
            if recovery_mode:
                successful_recoveries += 1
                print(f"✓ Frame {frame_count}: RECOVERED")
        else:
            detection_found = False
    
    # PHASE 3: UPDATE
    if tracking_initialized and detection_found:
        ekf.update(detected_x, detected_y)
        lost_frames = 0
        tracking_active = True
        recovery_mode = False
        
        # Add to trajectory
        trajectory.append((int(ekf.state[0, 0]), int(ekf.state[1, 0])))
        if len(trajectory) > max_trajectory_length:
            trajectory.pop(0)
    
    elif tracking_initialized and not detection_found:
        lost_frames += 1
        
        if lost_frames > max_lost_frames:
            tracking_active = False
            if lost_frames == max_lost_frames + 1:
                print(f"\n❌ Frame {frame_count}: TRACKING LOST")
    
    # Get current state
    current_state = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0, 'ax': 0, 'ay': 0}
    if tracking_initialized:
        current_state = ekf.get_state()
    
    # VISUALIZATION
    display_frame = frame.copy()
    
    # Info panel
    panel_height = 220
    cv2.rectangle(display_frame, (0, 0), (frame.shape[1], panel_height), (40, 40, 40), -1)
    
    cv2.putText(display_frame, f"EXTENDED KALMAN FILTER (EKF)", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
    
    cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Status
    if not tracking_initialized:
        cv2.putText(display_frame, "Status: SEARCHING...", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        if recovery_mode:
            status_color = (255, 165, 0)
            status_text = "RECOVERY"
        else:
            status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
            status_text = "TRACKING" if tracking_active else "LOST"
        
        cv2.putText(display_frame, f"Status: {status_text}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    det_color = (0, 255, 0) if detection_found else (0, 165, 255)
    cv2.putText(display_frame, f"Detection: {'FOUND' if detection_found else 'NOT FOUND'}", 
               (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 2)
    
    det_rate = (total_detections/frame_count)*100 if frame_count > 0 else 0
    cv2.putText(display_frame, f"Rate: {det_rate:.1f}% ({total_detections}/{frame_count})", 
               (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # EKF State Information
    if tracking_initialized:
        speed = np.sqrt(current_state['vx']**2 + current_state['vy']**2)
        accel = np.sqrt(current_state['ax']**2 + current_state['ay']**2)
        
        cv2.putText(display_frame, f"Speed: {speed:.1f} px/f", (10, 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Accel: {accel:.2f} px/f² (EKF advantage!)", (10, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    
    # Draw trajectory
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            cv2.line(display_frame, trajectory[i-1], trajectory[i], (255, 100, 255), 2)
    
    # Draw detection (GREEN)
    if detection_found:
        x, y, w, h = detection_info['bbox']
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(display_frame, (detected_x, detected_y), 6, (0, 255, 0), -1)
    
    # Draw prediction (RED)
    if tracking_initialized and pred_x is not None:
        cv2.circle(display_frame, (pred_x, pred_y), 8, (0, 0, 255), 2)
        
        search_color = (0, 165, 255) if recovery_mode else (0, 255, 255)
        search_thickness = 3 if recovery_mode else 2
        cv2.rectangle(display_frame, (search_x1, search_y1), (search_x2, search_y2), 
                     search_color, search_thickness)
    
    # Draw velocity vector (CYAN)
    if tracking_initialized:
        speed = np.sqrt(current_state['vx']**2 + current_state['vy']**2)
        if speed > 1.0:
            arrow_end_x = int(current_state['x'] + current_state['vx'] * 5)
            arrow_end_y = int(current_state['y'] + current_state['vy'] * 5)
            cv2.arrowedLine(display_frame, (int(current_state['x']), int(current_state['y'])), 
                           (arrow_end_x, arrow_end_y), (255, 255, 0), 2, tipLength=0.3)
    
    # Draw acceleration vector (MAGENTA)
    if tracking_initialized:
        accel = np.sqrt(current_state['ax']**2 + current_state['ay']**2)
        if accel > 0.5:
            accel_end_x = int(current_state['x'] + current_state['ax'] * 20)
            accel_end_y = int(current_state['y'] + current_state['ay'] * 20)
            cv2.arrowedLine(display_frame, (int(current_state['x']), int(current_state['y'])), 
                           (accel_end_x, accel_end_y), (255, 0, 255), 2, tipLength=0.4)
    
    cv2.putText(display_frame, "GREEN=Detection | RED=Predict | CYAN=Velocity | MAGENTA=Accel", 
               (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    if show_mask:
        mask = detect_white_objects(frame)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display_frame = cv2.addWeighted(display_frame, 0.7, mask_colored, 0.3, 0)
    
    cv2.imshow("EKF White Car Tracking", display_frame)
    
    if frame_count % 20 == 0:
        status_str = 'ACTIVE' if tracking_active else ('INIT' if not tracking_initialized else 'LOST')
        print(f"F{frame_count}: Det={'✓' if detection_found else '✗'} | Track={status_str} | Rate={det_rate:.1f}%")
    
    # KEY HANDLING
    key = cv2.waitKey(30) & 0xFF
    
    if key == 27 or key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)
    elif key == ord('d'):
        show_mask = not show_mask
    elif key == ord('r'):
        tracking_initialized = False
        tracking_active = False
        ekf = None
        lost_frames = 0
        trajectory = []

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("EKF TRACKING COMPLETE")
print("=" * 70)
print(f"Frames: {frame_count}")
print(f"Detections: {total_detections} ({det_rate:.1f}%)")
print(f"Recoveries: {successful_recoveries}/{recovery_attempts}")
print("=" * 70)