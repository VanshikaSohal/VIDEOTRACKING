"""
KALMAN FILTER PERSON TRACKER WITH OCCLUSION RECOVERY
Features:
- GREEN BOX = Detection (when person is visible)
- RED BOX = Kalman Prediction (when person is visible)
- NO BOXES = When person is hidden (occluded) - only shows "NOT FOUND"
- AUTOMATIC RECOVERY = Resumes tracking when person reappears
"""

import cv2
import numpy as np

class StandardKalmanFilter:
    def __init__(self, x, y, dt=1.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.dt = dt
        
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.008
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.5
        
        self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        
        self.prev_x = [x] * 2
        self.prev_y = [y] * 2
    
    def predict(self):
        prediction = self.kf.predict()
        pred_x = float(prediction[0][0])
        pred_y = float(prediction[1][0])
        
        vx = float(prediction[2][0])
        vy = float(prediction[3][0])
        pred_x = pred_x + vx * 0.7
        pred_y = pred_y + vy * 0.7

        self.prev_x.append(pred_x)
        self.prev_y.append(pred_y)
        self.prev_x.pop(0)
        self.prev_y.pop(0)
        
        smooth_x = int(np.mean(self.prev_x))
        smooth_y = int(np.mean(self.prev_y))
        
        return smooth_x, smooth_y
    
    def update(self, x, y):
        measurement = np.array([[x], [y]], np.float32)
        self.kf.correct(measurement)
    
    def get_state(self):
        state = self.kf.statePost
        return {
            'x': float(state[0][0]),
            'y': float(state[1][0]),
            'vx': float(state[2][0]),
            'vy': float(state[3][0])
        }


# ============================================================================
# OCCLUSION RECOVERY CLASS
# This handles the case when a person hides temporarily and then reappears
# ============================================================================
class OcclusionRecovery:
    """
    Handles occlusion recovery for person tracking
    
    When a person is hidden (occluded), this class:
    1. Stores the last known position and trajectory
    2. Continues predicting where the person should be (INTERNALLY - not shown on screen)
    3. When detections reappear near predicted position, re-establishes tracking
    4. While occluded, shows "NOT FOUND" instead of boxes
    """
    
    def __init__(self, max_occlusion_frames=90):
        """
        Initialize occlusion recovery
        
        Args:
            max_occlusion_frames: Maximum frames to maintain prediction during occlusion
                                 (default 90 = ~3 seconds at 30fps)
        """
        self.max_occlusion_frames = max_occlusion_frames
        self.in_occlusion = False
        self.occlusion_start_frame = 0
        self.last_known_position = None
        self.last_known_velocity = None
        self.predicted_positions = []  # Internal prediction history (not displayed)
        
    def start_occlusion(self, frame_num, position, velocity):
        """
        Called when person is first lost from detection
        Person is now HIDDEN - no boxes will be shown
        
        Args:
            frame_num: Current frame number
            position: Last known (x, y) position
            velocity: Last known (vx, vy) velocity
        """
        if not self.in_occlusion:
            self.in_occlusion = True
            self.occlusion_start_frame = frame_num
            self.last_known_position = position
            self.last_known_velocity = velocity
            self.predicted_positions = [position]
            
    def update_prediction(self, predicted_pos):
        """
        Store predicted position during occlusion (INTERNAL only - not displayed)
        
        Args:
            predicted_pos: Current (x, y) prediction from Kalman filter
        """
        if self.in_occlusion:
            self.predicted_positions.append(predicted_pos)
            # Keep only last 30 predictions to avoid memory buildup
            if len(self.predicted_positions) > 30:
                self.predicted_positions.pop(0)
    
    def check_recovery(self, frame_num, detection_center, search_radius=250):
        """
        Check if a detection matches our internal occlusion prediction
        
        This is the KEY RECOVERY LOGIC:
        - While person is hidden, we predict internally (not shown)
        - When a detection appears near where we predicted
        - We re-establish tracking and show boxes again
        
        Args:
            frame_num: Current frame number
            detection_center: (x, y) center of detected person
            search_radius: How far from prediction to accept detection (pixels)
            
        Returns:
            True if this detection matches our prediction (recovery successful)
        """
        if not self.in_occlusion:
            return False
        
        # Check if we've been occluded too long
        occlusion_duration = frame_num - self.occlusion_start_frame
        if occlusion_duration > self.max_occlusion_frames:
            return False
        
        # Get latest predicted position (internal)
        if not self.predicted_positions:
            return False
            
        pred_x, pred_y = self.predicted_positions[-1]
        det_x, det_y = detection_center
        
        # Calculate distance between detection and internal prediction
        distance = np.sqrt((det_x - pred_x)**2 + (det_y - pred_y)**2)
        
        # If detection is close to prediction, it's likely the same person returning
        # This is the RECOVERY TRIGGER - boxes will appear again
        if distance < search_radius:
            return True
        
        return False
    
    def end_occlusion(self):
        """
        Called when tracking is successfully re-established
        Person is now VISIBLE again - boxes will be shown
        """
        self.in_occlusion = False
        self.predicted_positions = []
        
    def get_occlusion_duration(self, frame_num):
        """Get how many frames person has been occluded"""
        if not self.in_occlusion:
            return 0
        return frame_num - self.occlusion_start_frame
    
    def is_occluded(self):
        """Check if currently in occlusion state (person is HIDDEN)"""
        return self.in_occlusion
    
    def reset(self):
        """Reset occlusion state"""
        self.in_occlusion = False
        self.occlusion_start_frame = 0
        self.last_known_position = None
        self.last_known_velocity = None
        self.predicted_positions = []


# ============================================================================
# Detection functions
# ============================================================================

def detect_person_motion(frame, prev_frame):
    """Improved motion detection"""
    if prev_frame is None:
        return []
    
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray1 = cv2.GaussianBlur(gray1, (11, 11), 0)
    gray2 = cv2.GaussianBlur(gray2, (11, 11), 0)
    
    diff = cv2.absdiff(gray1, gray2)
    
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area < 1500 or area > 25000:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        aspect_ratio = h / float(w) if w > 0 else 0
        if h > 45 and 1.5 < aspect_ratio < 3.5:
            cx, cy = x + w//2, y + h//2
            detections.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area
            })
    
    return detections

def detect_person_hog(frame):
    """HOG detection with better parameters"""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    try:
        boxes, weights = hog.detectMultiScale(
            frame,
            winStride=(4,4),
            padding=(8, 8),
            scale=1.05,
            hitThreshold=0.5
        )
        
        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            cx, cy = x + w//2, y + h//2
            detections.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': w * h,
                'confidence': float(weight)
            })
        
        return detections
    except:
        return []

def detect_background_subtraction(frame, bg_subtractor):
    """Background subtraction for moving objects"""
    
    fg_mask = bg_subtractor.apply(frame, learningRate=0.005)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area < 200 or area > 20000:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        aspect_ratio = h / float(w) if w > 0 else 0
        if h > 50 and 1.0 < aspect_ratio < 5.0:
            cx, cy = x + w//2, y + h//2
            detections.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area
            })
    
    return detections

def combine_detections(all_dets):
    """Merge overlapping detections"""
    if not all_dets:
        return []
    
    merged = []
    used = set()
    
    for i, det1 in enumerate(all_dets):
        if i in used:
            continue
        
        similar = [det1]
        cx1, cy1 = det1['center']
        
        for j, det2 in enumerate(all_dets[i+1:], i+1):
            if j in used:
                continue
            
            cx2, cy2 = det2['center']
            dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            
            if dist < 60:
                similar.append(det2)
                used.add(j)
        
        if similar:
            avg_cx = int(np.mean([d['center'][0] for d in similar]))
            avg_cy = int(np.mean([d['center'][1] for d in similar]))
            
            largest = max(similar, key=lambda d: d['area'])
            
            merged.append({
                'center': (avg_cx, avg_cy),
                'bbox': largest['bbox'],
                'area': np.mean([d['area'] for d in similar])
            })
    
    return sorted(merged, key=lambda d: d['area'], reverse=True)

def smooth_bbox(bbox, history, max_size=4):
    """Smooth bounding box using history with size stabilization"""
    if bbox is None:
        return None
    
    history.append(bbox)
    if len(history) > max_size:
        history.pop(0)
    
    if len(history) >= 2:
        x_coords = [b[0] for b in history]
        y_coords = [b[1] for b in history]
        w_coords = [b[2] for b in history]
        h_coords = [b[3] for b in history]
        
        smooth_x = int(np.median(x_coords))
        smooth_y = int(np.median(y_coords))
        smooth_w = int(np.mean(w_coords))
        smooth_h = int(np.mean(h_coords))
        
        smooth_w = max(70, min(smooth_w, 130))
        smooth_h = max(140, min(smooth_h, 250))
        
        return (smooth_x, smooth_y, smooth_w, smooth_h)
    else:
        return bbox


# ============================================================================
# MAIN TRACKING PROGRAM WITH OCCLUSION RECOVERY
# ============================================================================

print("="*70)
print("PERSON TRACKER WITH OCCLUSION RECOVERY")
print("="*70)
print("\n🎯 VISUALIZATION:")
print("   🟢 GREEN Box = DETECTION (actual measurement)")
print("   🔴 RED Box = KALMAN PREDICTION")
print("   ❌ NOT FOUND = Person is HIDDEN (occluded)")
print("\n✨ OCCLUSION RECOVERY:")
print("   - When person hides: Shows 'NOT FOUND' (no boxes)")
print("   - Internally tracks predicted position (not shown)")
print("   - Automatically resumes tracking when person reappears")
print("   - Works for up to 90 frames (~3 seconds) of occlusion")
print("\n🎮 CONTROLS: Q=Quit | SPACE=Pause | R=Reset")
print("="*70)

VIDEO_FILE = "occlusionvideo.mp4"

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"\n❌ Cannot open '{VIDEO_FILE}'")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dt = 1.0 / fps if fps > 0 else 0.033

print(f"\n📹 Video: {width}x{height} @ {fps:.1f} FPS\n")

# State
frame_count = 0
kf = None
tracking_init = False
lost_frames = 0
prev_frame = None

# Detection smoothing history
detection_history = []
max_history = 3

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Confidence tracking
consecutive_detections = 0
min_detection_confidence = 3

# OCCLUSION RECOVERY SYSTEM - This handles when person hides and reappears
occlusion_recovery = OcclusionRecovery(max_occlusion_frames=90)

# Stats
total_det = 0
occlusion_events = 0
recovery_events = 0

print("🎬 Starting detection...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    det_found = False
    pred_x = pred_y = None
    detected_person = None
    detection_bbox = None
    prediction_bbox = None
    
    # PHASE 1: INITIAL DETECTION
    if not tracking_init:
        motion_dets = detect_person_motion(frame, prev_frame)
        hog_dets = detect_person_hog(frame)
        bg_dets = detect_background_subtraction(frame, bg_subtractor)
        
        all_dets = motion_dets + hog_dets + bg_dets
        all_dets = combine_detections(all_dets)
        
        if all_dets:
            detected_person = all_dets[0]
            det_found = True
            dx, dy = detected_person['center']
            
            raw_bbox = detected_person['bbox']
            detection_bbox = smooth_bbox(raw_bbox, detection_history, max_history)
            total_det += 1
            
            consecutive_detections += 1
            
            if consecutive_detections >= min_detection_confidence:
                kf = StandardKalmanFilter(dx, dy, dt)
                tracking_init = True
                
                print(f"✅ Frame {frame_count}: Person DETECTED at ({dx}, {dy}) - Tracking STARTED")
        else:
            consecutive_detections = 0
        
        if frame_count % 30 == 0 and not tracking_init:
            print(f"Frame {frame_count}: Searching...")
    
    # PHASE 2: TRACKING MODE WITH OCCLUSION RECOVERY
    else:
        # PREDICT (always predict internally, even during occlusion)
        pred_x, pred_y = kf.predict()
        pred_x = max(0, min(pred_x, width - 1))
        pred_y = max(0, min(pred_y, height - 1))
        
        # Update occlusion recovery with current prediction (INTERNAL - not displayed)
        # This keeps track of where person should be while hidden
        if occlusion_recovery.is_occluded():
            occlusion_recovery.update_prediction((pred_x, pred_y))
        
        state = kf.get_state()
        
        if detection_history:
            last_bbox = detection_history[-1]
            pred_w, pred_h = last_bbox[2], last_bbox[3]
        else:
            pred_w, pred_h = 85, 175
        
        # Prepare prediction bbox (will only be shown if NOT occluded)
        prediction_bbox = (
            int(pred_x - pred_w//2),
            int(pred_y - pred_h//2),
            pred_w,
            pred_h
        )
        
        # TRY TO DETECT
        motion_dets = detect_person_motion(frame, prev_frame)
        bg_dets = detect_background_subtraction(frame, bg_subtractor)
        
        hog_dets = []
        if frame_count % 5 == 0:
            hog_dets = detect_person_hog(frame)
        
        all_dets = motion_dets + hog_dets + bg_dets
        all_dets = combine_detections(all_dets)
        
        if all_dets:
            # Find closest to prediction
            best_det = None
            best_dist = float('inf')
            
            # OCCLUSION RECOVERY: If we're in occlusion mode, use wider search radius
            search_radius = 250 if occlusion_recovery.is_occluded() else 180
            
            for det in all_dets:
                dx, dy = det['center']
                dist = np.sqrt((dx - pred_x)**2 + (dy - pred_y)**2)
                
                if dist < search_radius and dist < best_dist:
                    best_dist = dist
                    best_det = det
            
            if best_det:
                detected_person = best_det
                det_found = True
                dx, dy = detected_person['center']
                
                raw_bbox = detected_person['bbox']
                detection_bbox = smooth_bbox(raw_bbox, detection_history, max_history)
                total_det += 1
                
                # OCCLUSION RECOVERY CHECK
                # If we were in occlusion and found a matching detection, recovery successful!
                if occlusion_recovery.is_occluded():
                    if occlusion_recovery.check_recovery(frame_count, (dx, dy)):
                        occlusion_duration = occlusion_recovery.get_occlusion_duration(frame_count)
                        print(f"🎯 Frame {frame_count}: RECOVERY! Person reappeared after {occlusion_duration} frames")
                        recovery_events += 1
                        occlusion_recovery.end_occlusion()
                
                # UPDATE Kalman with new measurement
                kf.update(dx, dy)
                
                lost_frames = 0
            else:
                lost_frames += 1
                
                # START OCCLUSION if we just lost detection
                if lost_frames == 1 and not occlusion_recovery.is_occluded():
                    state = kf.get_state()
                    occlusion_recovery.start_occlusion(
                        frame_count,
                        (state['x'], state['y']),
                        (state['vx'], state['vy'])
                    )
                    occlusion_events += 1
                    print(f"⚠️ Frame {frame_count}: Person HIDDEN - Occlusion recovery mode (no boxes shown)")
        else:
            lost_frames += 1
            
            # START OCCLUSION if we just lost detection
            if lost_frames == 1 and not occlusion_recovery.is_occluded():
                state = kf.get_state()
                occlusion_recovery.start_occlusion(
                    frame_count,
                    (state['x'], state['y']),
                    (state['vx'], state['vy'])
                )
                occlusion_events += 1
                print(f"⚠️ Frame {frame_count}: Person HIDDEN - Occlusion recovery mode (no boxes shown)")
        
        # Give up only after max occlusion time exceeded
        if lost_frames > occlusion_recovery.max_occlusion_frames:
            tracking_init = False
            detection_history.clear()
            consecutive_detections = 0
            occlusion_recovery.reset()
            print(f"\n❌ Frame {frame_count}: Tracking LOST (exceeded max occlusion time) - Restarting detection")
        elif occlusion_recovery.is_occluded() and lost_frames % 30 == 0:
            duration = occlusion_recovery.get_occlusion_duration(frame_count)
            print(f"🔍 Frame {frame_count}: Still HIDDEN ({duration} frames) - Waiting for person to reappear...")
    
    # ========================================================================
    # VISUALIZATION - CRITICAL: NO BOXES WHEN OCCLUDED (PERSON IS HIDDEN)
    # ========================================================================
    display = frame.copy()
    
    # Info panel
    panel_h = 150
    cv2.rectangle(display, (0, 0), (width, panel_h), (40, 40, 40), -1)
    
    cv2.putText(display, f"Frame: {frame_count} | Detections: {total_det}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Status with occlusion indicator
    if tracking_init:
        if occlusion_recovery.is_occluded():
            # PERSON IS HIDDEN - Show NOT FOUND status
            status = "NOT FOUND (Hidden)"
            status_color = (0, 0, 255)  # Red for not found
            occlusion_dur = occlusion_recovery.get_occlusion_duration(frame_count)
            cv2.putText(display, f"Hidden for: {occlusion_dur} frames", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # PERSON IS VISIBLE - Normal tracking
            status = "TRACKING"
            status_color = (0, 255, 0) if det_found else (0, 165, 255)
    else:
        status = "SEARCHING"
        status_color = (0, 165, 255)
    
    cv2.putText(display, f"Status: {status}", 
               (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Show recovery stats
    cv2.putText(display, f"Occlusions: {occlusion_events} | Recoveries: {recovery_events}", 
               (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # DRAW BOXES - ONLY IF NOT OCCLUDED (person is visible)
    if tracking_init and not occlusion_recovery.is_occluded():
        # RED BOX = Prediction (only shown when person is visible)
        if prediction_bbox:
            px, py, pw, ph = prediction_bbox
            cv2.rectangle(display, (px, py), (px+pw, py+ph), (0, 0, 255), 3)
            cv2.putText(display, "PRED", (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # GREEN BOX = Detection (only when detected and visible)
        if det_found and detection_bbox:
            dx, dy, dw, dh = detection_bbox
            cv2.rectangle(display, (dx, dy), (dx+dw, dy+dh), (0, 255, 0), 3)
            cv2.putText(display, "DETECT", (dx, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # If occluded, show big "NOT FOUND" message in center
    if tracking_init and occlusion_recovery.is_occluded():
        text = "NOT FOUND - PERSON HIDDEN"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        cv2.putText(display, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    cv2.imshow("Person Tracker - Occlusion Recovery", display)
    
    prev_frame = frame.copy()
    
    # Controls
    key = cv2.waitKey(100) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)
    elif key == ord('r'):
        tracking_init = False
        kf = None
        lost_frames = 0
        prev_frame = None
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        detection_history = []
        consecutive_detections = 0
        occlusion_recovery.reset()
        print("\n🔄 RESET\n")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("FINAL STATISTICS")
print("="*70)
print(f"Total Frames: {frame_count}")
print(f"Total Detections: {total_det}")
print(f"Detection Rate: {(total_det/frame_count)*100:.1f}%")
print(f"Occlusion Events: {occlusion_events}")
print(f"Successful Recoveries: {recovery_events}")
if occlusion_events > 0:
    print(f"Recovery Success Rate: {(recovery_events/occlusion_events)*100:.1f}%")
print("="*70)