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
        
        # FIXED: Higher process noise to prevent velocity accumulation
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05  
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0  
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.5
        
        self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        
        self.prev_x = [x] * 2
        self.prev_y = [y] * 2
        self.velocity_reset_counter = 0  # NEW: Track when to reset velocity
    
    def predict(self):
        prediction = self.kf.predict()
        return int(prediction[0][0]), int(prediction[1][0])

    
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

def detect_person_motion(frame, prev_frame):
    if prev_frame is None:
        return []
    
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # FIXED: Stronger blur for distant motion
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)  # Changed from (11,11)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    
    diff = cv2.absdiff(gray1, gray2)
    # FIXED: Lower threshold for subtle motion when person is far
    _, thresh = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)  # Changed from 12
    
    # FIXED: Stronger morphology to connect body parts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Changed from (9,9)
    thresh = cv2.dilate(thresh, kernel, iterations=5)  # Changed from 4
    thresh = cv2.erode(thresh, kernel, iterations=3)   # Changed from 2
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # FIXED: Lower minimum area for distant person
        if area < 150 or area > 50000:  # Changed from 200
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w) if w > 0 else 0
        
        # FIXED: More lenient aspect ratio for varying distances
        if h > 15 and 1.2 < aspect_ratio < 4.5 and w > 10:  # More lenient
            cx, cy = x + w//2, y + h//2
            
            # Expand bbox
            padding_w = int(w * 0.6)  # Increased from 0.5
            padding_h = int(h * 0.4)  # Increased from 0.3
            x = max(0, x - padding_w)
            y = max(0, y - padding_h)
            w = w + 2 * padding_w
            h = h + 2 * padding_h
            
            detections.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': w * h
            })
    
    return detections

def detect_person_hog(frame):
    """HOG detection"""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    try:
        boxes, weights = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
            hitThreshold=0.0
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
    # FIXED: Higher learning rate for dynamic background
    fg_mask = bg_subtractor.apply(frame, learningRate=0.01)  # Changed from 0.005
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # Increased from (7,7)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # FIXED: Lower minimum for distant detection
        if area < 180 or area > 45000:  # Changed from 250
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w) if w > 0 else 0
        
        # FIXED: More lenient filtering
        if h > 20 and 1.2 < aspect_ratio < 5.0 and w > 15:  # More lenient
            cx, cy = x + w//2, y + h//2
            
            padding_w = int(w * 0.6)
            padding_h = int(w * 0.4)
            x = max(0, x - padding_w)
            y = max(0, y - padding_h)
            w = w + 2 * padding_w
            h = h + 2 * padding_h
            
            detections.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': w * h
            })
    
    return detections

def combine_detections(all_dets):
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
            
            # FIXED: Much larger merge distance
            if dist < 180:  # Changed from 120
                similar.append(det2)
                used.add(j)
        
        # Rest stays the same...
    return sorted(merged, key=lambda d: d['area'], reverse=True)

def smooth_bbox(bbox, history, max_size=4):
    """Smooth bounding box with aspect ratio enforcement"""
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
        smooth_w = int(np.median(w_coords))
        smooth_h = int(np.median(h_coords))
        
        # FIXED: Enforce aspect ratio 1.5:1 to 3:1 (height:width)
        aspect_ratio = smooth_h / float(smooth_w) if smooth_w > 0 else 2.0
        if aspect_ratio < 1.5:
            # Too wide - increase height
            smooth_h = int(smooth_w * 1.8)
        elif aspect_ratio > 3.0:
            # Too tall - increase width
            smooth_w = int(smooth_h / 2.2)
        
        # FIXED: Wider size range for distance variation
        smooth_w = max(40, min(smooth_w, 180))
        smooth_h = max(80, min(smooth_h, 320))
        
        return (smooth_x, smooth_y, smooth_w, smooth_h)
    else:
        return bbox

print("="*70)
print("FIXED PERSON TRACKER - FINAL VERSION")
print("="*70)
print("\n🎯 VISUALIZATION:")
print("   🟢 GREEN Box = DETECTION (actual measurement)")
print("   🔴 RED Box = KALMAN PREDICTION")
print("\n🎮 CONTROLS: Q=Quit | SPACE=Pause | R=Reset")
print("="*70)

VIDEO_FILE = "realvideo.mp4"

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"\n❌ Cannot open '{VIDEO_FILE}'")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dt = 1.0 / fps if fps > 0 else 0.033

print(f"\n📹 Video: {width}x{height} @ {fps:.1f} FPS\n")

frame_count = 0
kf = None
tracking_init = False
first_update = False
lost_frames = 0
prev_frame = None

detection_history = []
max_history = 3

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

consecutive_detections = 0
min_detection_confidence = 3

total_det = 0

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
        
        # FIXED: Filter out bad aspect ratios and tiny boxes
        all_dets = [d for d in all_dets if d['bbox'][2] > 25 and d['bbox'][3] > 50]
        
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: Searching... all_dets={len(all_dets)}")
        
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
                first_update = False
                
                print(f"✅ Frame {frame_count}: Person DETECTED at ({dx}, {dy}) - Tracking STARTED")
        else:
            consecutive_detections = 0
        
        if frame_count % 30 == 0 and not tracking_init:
            print(f"Frame {frame_count}: Still searching...")
    
    # PHASE 2: TRACKING MODE
    else:
        pred_x, pred_y = kf.predict()
        pred_x = max(0, min(pred_x, width - 1))
        pred_y = max(0, min(pred_y, height - 1))
        
        state = kf.get_state()
        
        # FIXED: Dynamic bbox sizing
        if detection_history and len(detection_history) >= 2:
            recent_boxes = detection_history[-3:] if len(detection_history) >= 3 else detection_history
            avg_w = int(np.mean([b[2] for b in recent_boxes]))
            avg_h = int(np.mean([b[3] for b in recent_boxes]))
            pred_w = max(50, min(avg_w, 180))
            pred_h = max(100, min(avg_h, 320))
        else:
            pred_w, pred_h = 90, 180
        
        prediction_bbox = (
            int(pred_x - pred_w//2),
            int(pred_y - pred_h//2),
            pred_w,
            pred_h
        )
        
        motion_dets = detect_person_motion(frame, prev_frame)
        bg_dets = detect_background_subtraction(frame, bg_subtractor)
        
        hog_dets = []
        if frame_count % 2 == 0:
            hog_dets = detect_person_hog(frame)
        
        all_dets = motion_dets + hog_dets + bg_dets
        all_dets = combine_detections(all_dets)
        
        # FIXED: Filter bad detections
        all_dets = [d for d in all_dets if d['bbox'][2] > 25 and d['bbox'][3] > 50]
        
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: pred=({pred_x},{pred_y}) det={det_found} lost={lost_frames} all_dets={len(all_dets)}")
        
        # In tracking mode section, change this:
    if all_dets:
        best_det = None
        best_dist = float('inf')
    
    # FIXED: Adaptive search radius based on lost frames
        search_radius = 250 + (lost_frames * 15)  # Grows with uncertainty
        search_radius = min(search_radius, 600)   # Cap at 600 pixels
    
        for det in all_dets:
           dx, dy = det['center']
        
        # FIXED: Pure Euclidean distance - no asymmetric weighting
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
                area = detected_person['area']

                if area < 3000:        # far
                   kf.kf.measurementNoiseCov = np.eye(2) * 12.0
                elif area < 8000:      # mid
                   kf.kf.measurementNoiseCov = np.eye(2) * 6.0
                else:                  # near
                  kf.kf.measurementNoiseCov = np.eye(2) * 2.5

                kf.update(dx, dy)
                lost_frames = 0
           else:
                lost_frames += 1
        else:
            lost_frames += 1
        
        # FIXED: Force correction if many detections but none matched
        if len(all_dets) > 2 and lost_frames > 5:
            nearest_det = min(all_dets, key=lambda d: np.sqrt((d['center'][0]-pred_x)**2 + (d['center'][1]-pred_y)**2))
            nearest_dist = np.sqrt((nearest_det['center'][0]-pred_x)**2 + (nearest_det['center'][1]-pred_y)**2)
            
            if nearest_dist < 350:
                detected_person = nearest_det
                det_found = True
                dx, dy = nearest_det['center']
                
                raw_bbox = nearest_det['bbox']
                detection_bbox = smooth_bbox(raw_bbox, detection_history, max_history)
                total_det += 1
                
                kf.update(dx, dy)
                lost_frames = 0
                print(f"🔄 Frame {frame_count}: Force-corrected to nearest detection (dist={nearest_dist:.0f})")
        
        if lost_frames > 45:
            tracking_init = False
            detection_history.clear()
            consecutive_detections = 0
            print(f"\n⚠️ Frame {frame_count}: Tracking LOST after {lost_frames} frames - Restarting")
        elif lost_frames > 0 and lost_frames % 20 == 0:
            print(f"⚠️ Frame {frame_count}: No detection for {lost_frames} frames (predicting...)")
    
    # VISUALIZATION
    display = frame.copy()
    
    panel_h = 120
    cv2.rectangle(display, (0, 0), (width, panel_h), (40, 40, 40), -1)
    
    cv2.putText(display, f"Frame: {frame_count} | Detections: {total_det}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    status = "TRACKING" if tracking_init else "SEARCHING"
    status_color = (0, 255, 0) if det_found else (0, 165, 255)
    cv2.putText(display, f"Status: {status}", 
               (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    if tracking_init and lost_frames > 0:
        cv2.putText(display, f"Lost: {lost_frames} frames", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    if tracking_init:
        if prediction_bbox:
            px, py, pw, ph = prediction_bbox
            cv2.rectangle(display, (px, py), (px+pw, py+ph), (0, 0, 255), 3)
            cv2.putText(display, "PRED", (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if det_found and detection_bbox:
            dx, dy, dw, dh = detection_bbox
            cv2.rectangle(display, (dx, dy), (dx+dw, dy+dh), (0, 255, 0), 3)
            cv2.putText(display, "DETECT", (dx, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Person Tracker", display)
    
    prev_frame = frame.copy()
    
    key = cv2.waitKey(100) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)
    elif key == ord('r'):
        tracking_init = False
        kf = None
        first_update = False
        lost_frames = 0
        prev_frame = None
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        detection_history = []
        consecutive_detections = 0
        print("\n🔄 RESET\n")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("FINAL STATISTICS")
print("="*70)
print(f"Total Frames: {frame_count}")
print(f"Total Detections: {total_det}")
print(f"Detection Rate: {(total_det/frame_count)*100:.1f}%")
print("="*70)