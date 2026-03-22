import cv2
import numpy as np

# ============================================
# UNSCENTED KALMAN FILTER IMPLEMENTATION
# ============================================

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for non-linear motion tracking
    State: [x, y, vx, vy, ax, ay, turn_rate]
    """
    
    def __init__(self, x, y, dt=1.0):
        """
        Initialize UKF
        x, y: initial position
        dt: time step (frame interval)
        """
        self.dt = dt
        self.n_state = 7  # [x, y, vx, vy, ax, ay, turn_rate]
        self.n_meas = 2   # [x, y]
        
        # State vector [x, y, vx, vy, ax, ay, turn_rate]
        self.state = np.array([x, y, 0, 0, 0, 0, 0], dtype=np.float64)
        
        # State covariance matrix
        self.P = np.eye(self.n_state, dtype=np.float64) * 10.0
        
        # Process noise covariance
        self.Q = np.eye(self.n_state, dtype=np.float64)
        self.Q[0:2, 0:2] *= 0.1   # Position noise
        self.Q[2:4, 2:4] *= 0.5   # Velocity noise
        self.Q[4:6, 4:6] *= 1.0   # Acceleration noise
        self.Q[6, 6] = 0.01       # Turn rate noise
        
        # Measurement noise covariance
        self.R = np.eye(self.n_meas, dtype=np.float64) * 2.0
        
        # UKF parameters
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        
        self.lambda_ = self.alpha**2 * (self.n_state + self.kappa) - self.n_state
        self.n_sigma = 2 * self.n_state + 1
        
        # Weights for sigma points
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)
        
        self.Wm[0] = self.lambda_ / (self.n_state + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n_state + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1.0 / (2.0 * (self.n_state + self.lambda_))
            self.Wc[i] = 1.0 / (2.0 * (self.n_state + self.lambda_))
        
        # Initialize sigma points (will be computed in predict)
        self.sigma_points = None
        self.sigma_points_pred = None
    
    def generate_sigma_points(self):
        """Generate sigma points around current state"""
        sigma_points = np.zeros((self.n_sigma, self.n_state))
        
        # Compute square root of covariance matrix
        try:
            U = np.linalg.cholesky((self.n_state + self.lambda_) * self.P)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(self.P)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
            U = eigenvectors @ np.diag(np.sqrt(eigenvalues * (self.n_state + self.lambda_)))
        
        # Mean sigma point
        sigma_points[0] = self.state
        
        # Positive sigma points
        for i in range(self.n_state):
            sigma_points[i + 1] = self.state + U[:, i]
        
        # Negative sigma points
        for i in range(self.n_state):
            sigma_points[self.n_state + i + 1] = self.state - U[:, i]
        
        return sigma_points
    
    def state_transition(self, state):
        """
        Non-linear state transition function
        Models: constant velocity + acceleration + turning motion
        """
        x, y, vx, vy, ax, ay, turn_rate = state
        
        dt = self.dt
        
        # Update velocity with acceleration
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        
        # Apply turn rate (rotate velocity vector)
        if abs(turn_rate) > 1e-6:
            cos_theta = np.cos(turn_rate * dt)
            sin_theta = np.sin(turn_rate * dt)
            
            vx_rotated = vx_new * cos_theta - vy_new * sin_theta
            vy_rotated = vx_new * sin_theta + vy_new * cos_theta
            
            vx_new = vx_rotated
            vy_new = vy_rotated
        
        # Update position with velocity
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt
        
        # Acceleration and turn rate decay (damping)
        ax_new = ax * 0.95
        ay_new = ay * 0.95
        turn_rate_new = turn_rate * 0.9
        
        return np.array([x_new, y_new, vx_new, vy_new, ax_new, ay_new, turn_rate_new])
    
    def measurement_function(self, state):
        """Extract measurement from state (just x, y position)"""
        return state[0:2]
    
    def predict(self):
        """
        Prediction step of UKF
        Returns: predicted (x, y) position
        """
        # Generate sigma points
        self.sigma_points = self.generate_sigma_points()
        
        # Propagate sigma points through state transition
        self.sigma_points_pred = np.zeros_like(self.sigma_points)
        for i in range(self.n_sigma):
            self.sigma_points_pred[i] = self.state_transition(self.sigma_points[i])
        
        # Compute predicted state (weighted mean)
        self.state = np.zeros(self.n_state)
        for i in range(self.n_sigma):
            self.state += self.Wm[i] * self.sigma_points_pred[i]
        
        # Compute predicted covariance
        self.P = np.copy(self.Q)
        for i in range(self.n_sigma):
            diff = self.sigma_points_pred[i] - self.state
            self.P += self.Wc[i] * np.outer(diff, diff)
        
        # Return predicted position
        pred_x = int(self.state[0])
        pred_y = int(self.state[1])
        
        return pred_x, pred_y
    
    def update(self, measured_x, measured_y):
        """
        Update step of UKF
        measured_x, measured_y: observed position
        """
        # Check if predict was called
        if self.sigma_points_pred is None:
            raise RuntimeError("Must call predict() before update()")
        
        # Measurement vector
        z = np.array([measured_x, measured_y], dtype=np.float64)
        
        # Transform sigma points to measurement space
        sigma_measurements = np.zeros((self.n_sigma, self.n_meas))
        for i in range(self.n_sigma):
            sigma_measurements[i] = self.measurement_function(self.sigma_points_pred[i])
        
        # Predicted measurement (weighted mean)
        z_pred = np.zeros(self.n_meas)
        for i in range(self.n_sigma):
            z_pred += self.Wm[i] * sigma_measurements[i]
        
        # Innovation covariance
        S = np.copy(self.R)
        for i in range(self.n_sigma):
            diff = sigma_measurements[i] - z_pred
            S += self.Wc[i] * np.outer(diff, diff)
        
        # Cross-covariance
        Pxz = np.zeros((self.n_state, self.n_meas))
        for i in range(self.n_sigma):
            state_diff = self.sigma_points_pred[i] - self.state
            meas_diff = sigma_measurements[i] - z_pred
            Pxz += self.Wc[i] * np.outer(state_diff, meas_diff)
        
        # Kalman gain
        try:
            K = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # If inversion fails, use pseudo-inverse
            K = Pxz @ np.linalg.pinv(S)
        
        # Update state
        innovation = z - z_pred
        self.state = self.state + K @ innovation
        
        # Update covariance
        self.P = self.P - K @ S @ K.T
    
    def get_state(self):
        """Get current state"""
        return self.state.copy()
    
    def get_velocity(self):
        """Get velocity magnitude"""
        vx, vy = self.state[2], self.state[3]
        return np.sqrt(vx**2 + vy**2)

# ============================================
# WHITE OBJECT DETECTION
# ============================================

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
                
                valid_candidates.append({
                    'contour': cnt,
                    'center': (cx, cy),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'quality_score': area
                })
    
    valid_candidates.sort(key=lambda d: d['quality_score'], reverse=True)
    return valid_candidates

# ============================================
# MAIN PROGRAM
# ============================================

print("=" * 70)
print("WHITE CAR TRACKING WITH UNSCENTED KALMAN FILTER (UKF)")
print("=" * 70)
print("\nUKF Features:")
print("  • 7-state model: [x, y, vx, vy, ax, ay, turn_rate]")
print("  • Handles non-linear motion (acceleration, turning)")
print("  • More accurate for complex vehicle dynamics")
print("\nControls: Q=Quit | SPACE=Pause | D=Mask | R=Reset")
print("=" * 70)

object_specs = {
    'color': 'white',
    'min_area': 3000,
    'max_area': 50000,
    'aspect_ratio': (0.3, 4.0),
    'region_of_interest': None
}

cap = cv2.VideoCapture("realcurvedvideo.mp4")
if not cap.isOpened():
    print("\n❌ ERROR: Cannot open video 'challenge.mp4'")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

dt = 1.0 / fps if fps > 0 else 1.0

print(f"\n📹 VIDEO: {frame_width}x{frame_height} @ {fps:.1f} FPS")
print(f"Time step (dt): {dt:.3f} seconds\n")

# Tracking variables
frame_count = 0
ukf = None
tracking_initialized = False
tracking_active = False
lost_frames = 0
max_lost_frames = 30
recovery_threshold = 15
recovery_mode = False
first_detection_frame = True  # NEW: Track first detection

# Statistics
total_detections = 0
total_predictions = 0
recovery_attempts = 0
successful_recoveries = 0

# Visualization
show_mask = False

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"\n🏁 Video ended")
        break
    
    frame_count += 1
    
    # Reset per-frame variables
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
            
            # Initialize UKF
            ukf = UnscentedKalmanFilter(detected_x, detected_y, dt=dt)
            tracking_initialized = True
            tracking_active = True
            first_detection_frame = True  # Mark this as first detection
            
            print(f"\n✓ CAR DETECTED at frame {frame_count}")
            print(f"  Position: ({detected_x}, {detected_y})")
            print(f"  UKF initialized with 7-state model [x,y,vx,vy,ax,ay,turn_rate]")
            print(f"  Skipping update on first frame (state already initialized)\n")
        else:
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: Searching for white car...")
    
    # PHASE 2: TRACKING WITH UKF PREDICTION
    else:
        # CRITICAL: Always call predict() first
        pred_x, pred_y = ukf.predict()
        total_predictions += 1
        
        pred_x = max(0, min(pred_x, frame_width - 1))
        pred_y = max(0, min(pred_y, frame_height - 1))
        
        # RECOVERY MODE
        if lost_frames >= recovery_threshold:
            recovery_mode = True
            if lost_frames == recovery_threshold:
                recovery_attempts += 1
                print(f"\n🔍 Frame {frame_count}: RECOVERY MODE")
            
            search_x1, search_y1 = 0, 0
            search_x2, search_y2 = frame_width, frame_height
        
        # NORMAL MODE
        else:
            recovery_mode = False
            
            velocity = ukf.get_velocity()
            base_search_size = max(200, int(velocity * 5))
            search_expansion = lost_frames * 25
            search_size = min(base_search_size + search_expansion, 600)
            
            search_x1 = max(0, pred_x - search_size)
            search_y1 = max(0, pred_y - search_size)
            search_x2 = min(frame_width, pred_x + search_size)
            search_y2 = min(frame_height, pred_y + search_size)
        
        # Search for car
        search_specs = object_specs.copy()
        search_specs['region_of_interest'] = (search_x1, search_y1, search_x2, search_y2)
        
        candidates = detect_white_car_candidates(frame, search_specs)
        
        # Find best candidate
        best_candidate = None
        min_dist = float('inf')
        
        for candidate in candidates:
            cx, cy = candidate['center']
            distance = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
            
            max_distance = 200 if not recovery_mode else float('inf')
            
            if distance < min_dist and distance < max_distance:
                min_dist = distance
                best_candidate = candidate
        
        if best_candidate is not None:
            detection_info = best_candidate
            detection_found = True
            detected_x, detected_y = detection_info['center']
            
            if detections_this_frame == 0:
                total_detections += 1
                detections_this_frame += 1
            
            if recovery_mode:
                successful_recoveries += 1
                print(f"✓ Frame {frame_count}: CAR RECOVERED")
    
    # PHASE 3: UKF UPDATE
    if tracking_initialized and detection_found:
        # IMPORTANT: Skip update on first detection frame
        # (UKF state is already initialized with the detection position)
        if first_detection_frame:
            first_detection_frame = False
            print(f"Frame {frame_count}: First frame - skipping update, UKF state initialized")
        else:
            # Update UKF with measurement
            ukf.update(detected_x, detected_y)
        
        lost_frames = 0
        tracking_active = True
        recovery_mode = False
    
    elif tracking_initialized and not detection_found:
        lost_frames += 1
        
        if lost_frames > max_lost_frames:
            tracking_active = False
            if lost_frames == max_lost_frames + 1:
                print(f"\n❌ Frame {frame_count}: TRACKING LOST")
    
    # Get current UKF state
    current_x, current_y = 0, 0
    current_vx, current_vy = 0, 0
    current_ax, current_ay = 0, 0
    turn_rate = 0
    
    if tracking_initialized and ukf is not None:
        state = ukf.get_state()
        current_x, current_y = state[0], state[1]
        current_vx, current_vy = state[2], state[3]
        current_ax, current_ay = state[4], state[5]
        turn_rate = state[6]
    
    # VISUALIZATION
    display_frame = frame.copy()
    
    # Info panel
    panel_height = 220
    cv2.rectangle(display_frame, (0, 0), (frame_width, panel_height), (40, 40, 40), -1)
    
    cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Status
    if not tracking_initialized:
        cv2.putText(display_frame, "Status: SEARCHING...", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        if recovery_mode:
            status_color = (255, 165, 0)
            status_text = "RECOVERY"
        else:
            status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
            status_text = "TRACKING" if tracking_active else "LOST"
        
        cv2.putText(display_frame, f"Status: {status_text}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Detection
    det_color = (0, 255, 0) if detection_found else (0, 165, 255)
    det_text = f"Detection: {'FOUND' if detection_found else 'NOT FOUND'}"
    cv2.putText(display_frame, det_text, (10, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 2)
    
    # Rate
    det_rate = (total_detections/frame_count)*100 if frame_count > 0 else 0
    cv2.putText(display_frame, f"Rate: {det_rate:.1f}% ({total_detections}/{frame_count})", 
               (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # UKF state info
    if tracking_initialized:
        speed = np.sqrt(current_vx**2 + current_vy**2)
        accel = np.sqrt(current_ax**2 + current_ay**2)
        
        cv2.putText(display_frame, f"Velocity: {speed:.1f} px/f | Accel: {accel:.2f}", 
                   (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Turn Rate: {turn_rate:.3f} rad/f | Lost: {lost_frames}", 
                   (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Position: ({int(current_x)}, {int(current_y)})", 
                   (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw detection (GREEN)
    if detection_found:
        x, y, w, h = detection_info['bbox']
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(display_frame, (detected_x, detected_y), 6, (0, 255, 0), -1)
    
    # Draw UKF prediction (RED)
    if tracking_initialized and pred_x is not None:
        cv2.circle(display_frame, (pred_x, pred_y), 8, (0, 0, 255), 2)
        cv2.putText(display_frame, "UKF", (pred_x + 12, pred_y - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Search region
        search_color = (0, 165, 255) if recovery_mode else (0, 255, 255)
        search_thickness = 3 if recovery_mode else 1
        cv2.rectangle(display_frame, (search_x1, search_y1), (search_x2, search_y2), 
                     search_color, search_thickness)
    
    # Draw velocity arrow
    if tracking_initialized:
        speed = np.sqrt(current_vx**2 + current_vy**2)
        if speed > 1.0:
            arrow_end_x = int(current_x + current_vx * 3)
            arrow_end_y = int(current_y + current_vy * 3)
            cv2.arrowedLine(display_frame, (int(current_x), int(current_y)), 
                           (arrow_end_x, arrow_end_y), (255, 255, 0), 2, tipLength=0.3)
    
    # Draw acceleration arrow (if significant)
    if tracking_initialized:
        accel = np.sqrt(current_ax**2 + current_ay**2)
        if accel > 0.5:
            accel_end_x = int(current_x + current_ax * 20)
            accel_end_y = int(current_y + current_ay * 20)
            cv2.arrowedLine(display_frame, (int(current_x), int(current_y)), 
                           (accel_end_x, accel_end_y), (255, 0, 255), 2, tipLength=0.4)
    
    cv2.putText(display_frame, "GREEN=Detection | RED=UKF | YELLOW=Velocity | MAGENTA=Accel", 
               (10, frame_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # Show mask if enabled
    if show_mask:
        mask = detect_white_objects(frame)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display_frame = cv2.addWeighted(display_frame, 0.7, mask_colored, 0.3, 0)
    
    cv2.imshow("UKF White Car Tracking", display_frame)
    
    # Terminal output
    if frame_count % 20 == 0:
        status = 'ACTIVE' if tracking_active else ('INIT' if not tracking_initialized else 'LOST')
        print(f"F{frame_count}: Det={'✓' if detection_found else '✗'} | "
              f"{status} | Rate={det_rate:.1f}%")
    
    # Sanity check
    if total_detections > frame_count:
        print(f"\n⚠️ BUG: detections ({total_detections}) > frames ({frame_count})")
        total_detections = frame_count
    
    # KEY HANDLING
    key = cv2.waitKey(30) & 0xFF
    
    if key == 27 or key == ord('q'):
        print("\n👋 Quitting...")
        break
    elif key == ord(' '):
        print("\n⏸ PAUSED")
        cv2.waitKey(0)
        print("▶ RESUMED")
    elif key == ord('d'):
        show_mask = not show_mask
    elif key == ord('r'):
        print("\n🔄 RESET")
        tracking_initialized = False
        ukf = None

cap.release()
cv2.destroyAllWindows()

# FINAL STATISTICS
print("\n" + "=" * 70)
print("FINAL STATISTICS")
print("=" * 70)
print(f"Frames: {frame_count}")
print(f"Detections: {total_detections} ({(total_detections/frame_count)*100:.1f}%)")
print(f"Predictions: {total_predictions}")
print(f"Recoveries: {successful_recoveries}/{recovery_attempts}")
print(f"Status: {'ACTIVE' if tracking_active else 'LOST'}")
print("=" * 70)