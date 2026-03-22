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
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        
        self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.statePost = self.kf.statePre.copy()
    
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
            'vy': float(state[3][0]),
            'speed': np.sqrt(float(state[2][0])**2 + float(state[3][0])**2)
        }

# Color-based detection
def detect_person_color(frame):
    """Detect person using color thresholds"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Skin tones
    lower_skin1 = np.array([0, 10, 60])
    upper_skin1 = np.array([20, 150, 255])
    mask_skin = cv2.inRange(hsv, lower_skin1, upper_skin1)
    
    lower_skin2 = np.array([160, 10, 60])
    upper_skin2 = np.array([180, 150, 255])
    mask_skin = cv2.bitwise_or(mask_skin, cv2.inRange(hsv, lower_skin2, upper_skin2))
    
    # Clothing: Brown/tan
    lower_brown = np.array([8, 30, 70])
    upper_brown = np.array([30, 220, 250])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Clothing: Dark (black, dark gray)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 100, 80])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Combine
    mask = cv2.bitwise_or(mask_skin, mask_brown)
    mask = cv2.bitwise_or(mask, mask_dark)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

# Motion-based detection
def detect_person_motion(frame, prev_frame):
    """Detect moving person"""
    if prev_frame is None:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray1 = cv2.GaussianBlur(gray1, (7, 7), 0)
    gray2 = cv2.GaussianBlur(gray2, (7, 7), 0)
    
    diff = cv2.absdiff(gray1, gray2)
    _, mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

# Find candidates
def find_person_candidates(frame, prev_frame):
    """Find person candidates"""
    color_mask = detect_person_color(frame)
    motion_mask = detect_person_motion(frame, prev_frame)
    
    # Use OR to be more permissive
    combined_mask = cv2.bitwise_or(color_mask, motion_mask)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > 20000:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        if h > 0 and w > 0:
            ar = w / h
            if 0.2 <= ar <= 0.95 and h > 50:
                cx, cy = x + w // 2, y + h // 2
                
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    compactness = 0.5
                
                if compactness > 0.15:
                    candidates.append({
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'score': area * compactness
                    })
    
    return sorted(candidates, key=lambda c: c['score'], reverse=True)

# Extract and compare templates
def extract_template(frame, bbox, size=(64, 64)):
    x, y, w, h = bbox
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return None
    
    template = frame[y:y+h, x:x+w].copy()
    return cv2.resize(template, size)

def template_match_score(t1, t2):
    if t1 is None or t2 is None:
        return 0.5
    
    try:
        hsv1 = cv2.cvtColor(t1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(t2, cv2.COLOR_BGR2HSV)
        
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [30, 32], [0, 180, 0, 256])
        
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    except:
        return 0.5

# Main program
print("="*70)
print("PERSON TRACKER - IMPROVED FOR REALVIDEO.MP4")
print("="*70)
print("\nMethods: Color + Motion Fusion")
print("Filter: Kalman (4-state: position + velocity)")
print("\nControls: Q=Quit | SPACE=Pause | R=Reset | D=Debug Masks")
print("="*70)

VIDEO_FILE = "realvideo.mp4"

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"ERROR: Cannot open {VIDEO_FILE}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dt = 1.0 / fps if fps > 0 else 1.0

print(f"\nVideo: {width}x{height} @ {fps:.1f} FPS")
print(f"Total Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}\n")

frame_count = 0
kf = None
tracking_init = False
tracking_active = False
lost_frames = 0
prev_frame = None
template = None
area_history = []
height_history = []
position_history = []
show_debug = False

total_det = 0
total_pred = 0
recoveries = 0

print("Starting detection...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    det_found = False
    
    # DETECTION PHASE
    if not tracking_init:
        candidates = find_person_candidates(frame, prev_frame)
        
        if candidates:
            det = candidates[0]
            cx, cy = det['center']
            kf = StandardKalmanFilter(cx, cy, dt)
            tracking_init = True
            tracking_active = True
            template = extract_template(frame, det['bbox'])
            area_history.append(det['area'])
            height_history.append(det['bbox'][3])
            det_found = True
            total_det += 1
            
            print(f"[Frame {frame_count}] Person detected at ({cx}, {cy})")
            print(f"  Area: {det['area']:.0f}px, Height: {det['bbox'][3]}px\n")
        
        elif frame_count % 30 == 0:
            print(f"[Frame {frame_count}] Searching...")
    
    # TRACKING PHASE
    else:
        pred_x, pred_y = kf.predict()
        pred_x = max(0, min(pred_x, width - 1))
        pred_y = max(0, min(pred_y, height - 1))
        total_pred += 1
        
        # Search region
        if lost_frames < 10:
            state = kf.get_state()
            search_dist = max(80, int(state['speed'] * 6))
        else:
            search_dist = 200
        
        sx1 = max(0, pred_x - search_dist)
        sy1 = max(0, pred_y - search_dist)
        sx2 = min(width, pred_x + search_dist)
        sy2 = min(height, pred_y + search_dist)
        
        # Find candidates in search region
        candidates = find_person_candidates(frame, prev_frame)
        
        # Filter by search region and match
        best_match = None
        best_score = -1
        
        for cand in candidates:
            cx, cy = cand['center']
            if not (sx1 <= cx <= sx2 and sy1 <= cy <= sy2):
                continue
            
            # Distance score
            dist = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
            dist_score = max(0, 1.0 - dist / max(search_dist, 1))
            
            # Template score
            cand_template = extract_template(frame, cand['bbox'])
            temp_score = template_match_score(template, cand_template)
            
            # Height consistency
            if height_history:
                avg_h = np.mean(height_history[-5:])
                h_diff = abs(cand['bbox'][3] - avg_h) / avg_h
                height_score = max(0, 1.0 - h_diff)
            else:
                height_score = 0.5
            
            score = dist_score * 0.4 + temp_score * 0.4 + height_score * 0.2
            
            if score > best_score:
                best_score = score
                best_match = cand
        
        # Accept match if good enough
        min_score = 0.25 if lost_frames > 5 else 0.35
        
        if best_match and best_score > min_score:
            cx, cy = best_match['center']
            kf.update(cx, cy)
            lost_frames = 0
            tracking_active = True
            det_found = True
            total_det += 1
            
            area_history.append(best_match['area'])
            height_history.append(best_match['bbox'][3])
            
            if len(area_history) > 15:
                area_history.pop(0)
            if len(height_history) > 15:
                height_history.pop(0)
            
            # Update template
            new_temp = extract_template(frame, best_match['bbox'])
            if new_temp is not None and template is not None:
                template = cv2.addWeighted(template, 0.7, new_temp, 0.3, 0)
            elif new_temp is not None:
                template = new_temp
        else:
            lost_frames += 1
            if lost_frames > 30:
                tracking_active = False
                if lost_frames == 31:
                    print(f"[Frame {frame_count}] Tracking LOST")
    
    # Get state for visualization
    if tracking_init:
        state = kf.get_state()
        cx, cy = int(state['x']), int(state['y'])
        vx, vy = state['vx'], state['vy']
        speed = state['speed']
    else:
        cx = cy = vx = vy = speed = 0
    
    # VISUALIZATION
    display = frame.copy()
    
    # Info panel
    panel_h = 100
    cv2.rectangle(display, (0, 0), (width, panel_h), (30, 30, 30), -1)
    
    cv2.putText(display, f"Person Tracker | Frame: {frame_count}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    status_text = "LOST"
    status_color = (0, 0, 255)
    if tracking_init:
        if tracking_active:
            status_text = "TRACKING"
            status_color = (0, 255, 0)
        else:
            status_text = "LOST (waiting)"
            status_color = (0, 165, 255)
    else:
        status_text = "SEARCHING"
        status_color = (255, 165, 0)
    
    cv2.putText(display, f"Status: {status_text}", (10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
    
    det_rate = (total_det / frame_count) * 100 if frame_count > 0 else 0
    cv2.putText(display, f"Detection Rate: {det_rate:.1f}% | Lost Frames: {lost_frames}", 
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    if tracking_init and det_found:
        cv2.putText(display, f"Speed: {speed:.1f} px/f", 
                    (width - 250, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw search region
    if tracking_init and not det_found:
        cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)
    
    # Draw position history
    if len(position_history) > 1:
        for i in range(len(position_history) - 1):
            cv2.line(display, position_history[i], position_history[i+1], (255, 0, 255), 1)
    
    if tracking_init:
        position_history.append((cx, cy))
        if len(position_history) > 30:
            position_history.pop(0)
    
    # Draw predictions and detections
    if tracking_init and not det_found:
        cv2.circle(display, (pred_x, pred_y), 8, (0, 0, 255), 2)
    
    cv2.putText(display, "GREEN=Detect | RED=Predict | MAGENTA=Path", 
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    # Show debug masks
    if show_debug:
        color_mask = detect_person_color(frame)
        motion_mask = detect_person_motion(frame, prev_frame)
        
        color_mask_3ch = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        motion_mask_3ch = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        
        combined = cv2.hconcat([color_mask_3ch, motion_mask_3ch])
        cv2.imshow("Debug: Color | Motion", combined)
    
    cv2.imshow("Person Tracker", display)
    
    # Terminal output
    if frame_count % 30 == 0:
        print(f"[Frame {frame_count}] Det: {'Y' if det_found else 'N'} | Status: {status_text} | Rate: {det_rate:.1f}%")
    
    # Save frame
    prev_frame = frame.copy()
    
    # Key handling
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
        print("\nQuitting...")
        break
    elif key == ord(' '):
        print("PAUSED")
        cv2.waitKey(0)
    elif key == ord('r'):
        print("RESET")
        tracking_init = False
        kf = None
        template = None
        area_history = []
        height_history = []
        lost_frames = 0
    elif key == ord('d'):
        show_debug = not show_debug

cap.release()
cv2.destroyAllWindows()

# Final stats
print("\n" + "="*70)
print("FINAL STATISTICS")
print("="*70)
print(f"Total Frames: {frame_count}")
print(f"Detections: {total_det} ({(total_det/frame_count)*100:.1f}%)")
print(f"Predictions: {total_pred}")
print(f"Final Status: {'TRACKING' if tracking_active else 'LOST'}")
print("="*70)
