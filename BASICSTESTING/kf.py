import cv2
import numpy as np

# Detect White Objects Function
def detect_white_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # White color range in HSV
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([180, 30, 255])
    
    # Mask filters white color
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Additional cleanup
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    
    # Apply Gaussian blur to reduce noise
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask

# Enhanced Detection with Multiple Candidates
def detect_white_car_candidates(frame, object_specs):
    """
    Detect ALL white car candidates and return them sorted by quality
    """
    mask = detect_white_objects(frame)
    
    # Apply ROI if specified
    if object_specs.get('region_of_interest') is not None:
        x1, y1, x2, y2 = object_specs['region_of_interest']
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        mask = roi_mask
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Relaxed area constraints for partial occlusion
        min_area = object_specs['min_area'] * 0.4  # Allow 60% occluded
        max_area = object_specs.get('max_area', float('inf'))
        
        if area < min_area or area > max_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Relaxed aspect ratio for partial occlusion
        if h > 0:
            aspect_ratio = w / h
            min_ar, max_ar = object_specs['aspect_ratio']
            
            # Expand aspect ratio tolerance
            expanded_min_ar = min_ar * 0.5
            expanded_max_ar = max_ar * 1.5
            
            if expanded_min_ar <= aspect_ratio <= expanded_max_ar:
                cx = x + w // 2
                cy = y + h // 2
                
                # Calculate quality score
                quality_score = area  # Prefer larger objects
                
                valid_candidates.append({
                    'contour': cnt,
                    'center': (cx, cy),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'quality_score': quality_score
                })
    
    # Sort by quality score (area)
    valid_candidates.sort(key=lambda d: d['quality_score'], reverse=True)
    
    return valid_candidates

# Template Matching for Object Identity
def extract_object_template(frame, bbox, template_size=(64, 64)):
    """Extract a template from the object for identity tracking"""
    x, y, w, h = bbox
    
    # Ensure valid bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return None
    
    template = frame[y:y+h, x:x+w].copy()
    
    # Resize to standard size
    template = cv2.resize(template, template_size)
    
    return template

def compare_templates(template1, template2):
    """Compare two templates using histogram correlation"""
    if template1 is None or template2 is None:
        return 0.0
    
    # Convert to HSV for better color comparison
    hsv1 = cv2.cvtColor(template1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(template2, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    # Normalize
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Compare using correlation
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return similarity

# Create Kalman Filter Function
def create_kalman_filter(x, y):
    kf = cv2.KalmanFilter(4, 2)
    
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)
    
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], np.float32)
    
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    
    kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
    kf.statePost = kf.statePre.copy()
    
    return kf

def kalman_predict(kf):
    prediction = kf.predict()
    pred_x = int(prediction[0][0])
    pred_y = int(prediction[1][0])
    return pred_x, pred_y

def kalman_update(kf, x, y):
    measurement = np.array([[x], [y]], np.float32)
    kf.correct(measurement)

# ============================================
# MAIN PROGRAM
# ============================================

print("=" * 70)
print("ROBUST WHITE CAR DETECTION WITH KALMAN FILTER")
print("=" * 70)
print("\n🎯 FEATURES:")
print("  ✓ Handles multiple white cars (uses template matching)")
print("  ✓ Handles partial occlusion (relaxed constraints)")
print("  ✓ Handles scale changes (adaptive size)")
print("  ✓ Recovery mode for long occlusions")
print("  ✓ Out-of-frame detection")
print("  ✓ Wrong initial detection prevention")
print("\nOBJECT SPECIFICATIONS:")
print("  Color: White")
print("  Min Area: 3000 pixels (1200 when partially occluded)")
print("  Max Area: 50000 pixels")
print("  Aspect Ratio: 0.15 to 6.0 (adaptive)")
print("\nCONTROLS:")
print("  Q/ESC - Quit")
print("  SPACE - Pause/Resume")
print("  D - Toggle detection mask")
print("  R - Reset tracking (select new car)")
print("  T - Show/hide template")
print("=" * 70)

# OBJECT SPECIFICATIONS
object_specs = {
    'color': 'white',
    'min_area': 3000,
    'max_area': 50000,
    'aspect_ratio': (0.3, 4.0),
    'region_of_interest': None
}

# Video capture
cap = cv2.VideoCapture("challenge.mp4")
if not cap.isOpened():
    print("\n❌ ERROR: Cannot open video file 'challenge.mp4'")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n📹 VIDEO INFO: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")
print("\nStarting detection...\n")

# Tracking variables
frame_count = 0
kf = None
tracking_initialized = False
tracking_active = False
lost_frames = 0
max_lost_frames = 45  # Increased tolerance
recovery_threshold = 15
recovery_mode = False

# Object identity tracking
object_template = None
template_update_counter = 0
template_update_interval = 20

# Area tracking for scale adaptation
tracked_areas = []
max_area_history = 10

# Statistics
total_detections = 0
total_predictions = 0
recovery_attempts = 0
successful_recoveries = 0
template_matches = 0
out_of_frame_warnings = 0

# Visualization
show_mask = False
show_template = False

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"\n🏁 Video ended at frame {frame_count}")
        break
    
    frame_count += 1
    
    # CRITICAL: Reset all detection variables for THIS frame
    detection_found = False
    detected_x, detected_y = None, None
    pred_x, pred_y = None, None
    detection_info = None
    candidates = []
    search_x1, search_y1, search_x2, search_y2 = 0, 0, 0, 0
    
    # Debug counter to ensure we only count detection ONCE per frame
    detections_this_frame = 0
    
    # PHASE 1: INITIAL DETECTION (First frame only)
    if not tracking_initialized:
        candidates = detect_white_car_candidates(frame, object_specs)
        
        if len(candidates) > 0:
            # If multiple candidates, let user choose or pick largest
            if len(candidates) > 1:
                print(f"\n⚠️  Frame {frame_count}: Found {len(candidates)} white car candidates")
                print(f"   Auto-selecting largest candidate (area={candidates[0]['area']})")
            
            detection_info = candidates[0]
            detection_found = True
            detected_x, detected_y = detection_info['center']
            
            # COUNT DETECTION ONLY ONCE
            if detections_this_frame == 0:
                total_detections += 1
                detections_this_frame += 1
            
            # Initialize tracking
            kf = create_kalman_filter(detected_x, detected_y)
            tracking_initialized = True
            tracking_active = True
            
            # Extract template for identity tracking
            object_template = extract_object_template(frame, detection_info['bbox'])
            tracked_areas.append(detection_info['area'])
            
            print(f"\n✓ WHITE CAR DETECTED at frame {frame_count}")
            print(f"  Position: ({detected_x}, {detected_y})")
            print(f"  Area: {detection_info['area']} pixels")
            print(f"  Template extracted for identity tracking\n")
        else:
            # NO DETECTION - explicitly set to False
            detection_found = False
            print(f"Frame {frame_count}: No white car candidates found")
    
    # PHASE 2: TRACKING MODE
    else:
        pred_x, pred_y = kalman_predict(kf)
        total_predictions += 1
        
        pred_x = max(0, min(pred_x, frame.shape[1] - 1))
        pred_y = max(0, min(pred_y, frame.shape[0] - 1))
        
        # Check if prediction is near frame boundary
        boundary_margin = 50
        near_boundary = (pred_x < boundary_margin or pred_x > frame_width - boundary_margin or
                        pred_y < boundary_margin or pred_y > frame_height - boundary_margin)
        
        # RECOVERY MODE
        if lost_frames >= recovery_threshold:
            recovery_mode = True
            recovery_attempts += 1
            
            search_x1, search_y1 = 0, 0
            search_x2, search_y2 = frame.shape[1], frame.shape[0]
            
            if lost_frames == recovery_threshold:
                print(f"\n🔍 Frame {frame_count}: RECOVERY MODE (lost {lost_frames} frames)")
        
        # NORMAL MODE
        else:
            recovery_mode = False
            
            # Adaptive search size based on velocity and history
            state = kf.statePost
            vx = float(state[2][0])
            vy = float(state[3][0])
            speed = np.sqrt(vx**2 + vy**2)
            
            # Larger search for fast-moving objects
            base_search_size = max(200, int(speed * 10))
            search_expansion = lost_frames * 25
            search_size = min(base_search_size + search_expansion, 600)
            
            search_x1 = max(0, pred_x - search_size)
            search_y1 = max(0, pred_y - search_size)
            search_x2 = min(frame.shape[1], pred_x + search_size)
            search_y2 = min(frame.shape[0], pred_y + search_size)
        
        # Update specs with search region
        search_specs = object_specs.copy()
        search_specs['region_of_interest'] = (search_x1, search_y1, search_x2, search_y2)
        
        # Adapt area constraints based on tracking history
        if len(tracked_areas) > 0:
            avg_area = np.mean(tracked_areas[-5:])  # Last 5 frames
            search_specs['min_area'] = int(avg_area * 0.3)  # Allow 70% size change
            search_specs['max_area'] = int(avg_area * 3.0)
        
        # Get all candidates
        candidates = detect_white_car_candidates(frame, search_specs)
        
        # Find best match using distance + template similarity
        best_candidate = None
        best_score = -float('inf')
        
        for candidate in candidates:
            cx, cy = candidate['center']
            distance = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
            
            # Distance score (closer is better)
            max_distance = 200 + (lost_frames * 15)
            if distance > max_distance and not recovery_mode:
                continue
            
            distance_score = 1.0 - (distance / max_distance)
            
            # Template similarity score
            if object_template is not None:
                candidate_template = extract_object_template(frame, candidate['bbox'])
                template_similarity = compare_templates(object_template, candidate_template)
            else:
                template_similarity = 0.5
            
            # Combined score
            combined_score = (distance_score * 0.4) + (template_similarity * 0.6)
            
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = candidate
        
        # Accept detection if score is good enough
        min_score_threshold = 0.3 if recovery_mode else 0.5
        
        if best_candidate is not None and best_score > min_score_threshold:
            detection_info = best_candidate
            detection_found = True
            detected_x, detected_y = detection_info['center']
            
            # COUNT DETECTION ONLY ONCE
            if detections_this_frame == 0:
                total_detections += 1
                detections_this_frame += 1
            
            template_matches += 1
            
            if recovery_mode:
                successful_recoveries += 1
                print(f"✓ Frame {frame_count}: CAR RECOVERED (score={best_score:.2f})")
        else:
            # NO DETECTION FOUND - explicitly set to False
            detection_found = False
            
            # Debug output for why detection failed
            if len(candidates) == 0:
                pass  # No candidates found in search region
            elif best_candidate is None:
                pass  # Candidates found but all too far from prediction
            else:
                # Score too low
                if frame_count % 20 == 0:  # Print occasionally
                    print(f"Frame {frame_count}: Detection rejected (score={best_score:.2f} < {min_score_threshold})")
    
    # PHASE 3: UPDATE
    if tracking_initialized and detection_found:
        kalman_update(kf, detected_x, detected_y)
        lost_frames = 0
        tracking_active = True
        recovery_mode = False
        
        # Update area history
        tracked_areas.append(detection_info['area'])
        if len(tracked_areas) > max_area_history:
            tracked_areas.pop(0)
        
        # Update template periodically
        template_update_counter += 1
        if template_update_counter >= template_update_interval:
            new_template = extract_object_template(frame, detection_info['bbox'])
            if new_template is not None:
                # Blend old and new template
                object_template = cv2.addWeighted(object_template, 0.3, new_template, 0.7, 0)
            template_update_counter = 0
    
    elif tracking_initialized and not detection_found:
        lost_frames += 1
        
        # Check if car likely left the frame
        if near_boundary and lost_frames > 10:
            out_of_frame_warnings += 1
            if out_of_frame_warnings == 1:
                print(f"\n⚠️  Frame {frame_count}: Car near frame boundary - may have exited")
        
        if lost_frames > max_lost_frames:
            tracking_active = False
            if lost_frames == max_lost_frames + 1:
                print(f"\n❌ Frame {frame_count}: TRACKING LOST (no detection for {max_lost_frames} frames)")
    
    # Get current state
    current_x, current_y = 0, 0
    current_vx, current_vy = 0, 0
    
    if tracking_initialized:
        state = kf.statePost
        current_x = float(state[0][0])
        current_y = float(state[1][0])
        current_vx = float(state[2][0])
        current_vy = float(state[3][0])
    
    # VISUALIZATION
    display_frame = frame.copy()
    
    # Info panel
    panel_height = 200
    cv2.rectangle(display_frame, (0, 0), (frame.shape[1], panel_height), (40, 40, 40), -1)
    
    cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Status
    if not tracking_initialized:
        cv2.putText(display_frame, "Status: SEARCHING...", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        if recovery_mode:
            status_color = (255, 165, 0)
            status_text = "RECOVERY MODE"
        else:
            status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
            status_text = "TRACKING" if tracking_active else "LOST"
        
        cv2.putText(display_frame, f"Status: {status_text}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Detection info with debug
    det_color = (0, 255, 0) if detection_found else (0, 165, 255)
    det_status = 'FOUND' if detection_found else 'NOT FOUND'
    cv2.putText(display_frame, f"Detection: {det_status}", 
               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 2)
    
    if detection_found:
        cv2.putText(display_frame, f"Candidates: {len(candidates)} | Area: {detection_info['area']}", 
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        # Show why detection failed
        reason = f"No candidates" if len(candidates) == 0 else f"{len(candidates)} candidates (none match)"
        cv2.putText(display_frame, reason, 
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    det_rate = (total_detections/frame_count)*100 if frame_count > 0 else 0
    
    # Show detection count for debugging
    cv2.putText(display_frame, f"Detection Rate: {det_rate:.1f}% ({total_detections}/{frame_count})", 
               (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if tracking_initialized:
        speed = np.sqrt(current_vx**2 + current_vy**2)
        cv2.putText(display_frame, f"Speed: {speed:.1f} px/f | Lost: {lost_frames}", 
                   (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw detection (GREEN)
    if detection_found:
        x, y, w, h = detection_info['bbox']
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(display_frame, (detected_x, detected_y), 6, (0, 255, 0), -1)
        cv2.putText(display_frame, "DETECTED", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw prediction (RED)
    if tracking_initialized and pred_x is not None:
        cv2.circle(display_frame, (pred_x, pred_y), 8, (0, 0, 255), 2)
        
        # Draw search region
        search_color = (0, 165, 255) if recovery_mode else (0, 255, 255)
        search_thickness = 3 if recovery_mode else 2
        cv2.rectangle(display_frame, (search_x1, search_y1), (search_x2, search_y2), 
                     search_color, search_thickness)
    
    # Draw velocity
    if tracking_initialized:
        speed = np.sqrt(current_vx**2 + current_vy**2)
        if speed > 1.0:
            arrow_end_x = int(current_x + current_vx * 5)
            arrow_end_y = int(current_y + current_vy * 5)
            cv2.arrowedLine(display_frame, (int(current_x), int(current_y)), 
                           (arrow_end_x, arrow_end_y), (255, 255, 0), 2, tipLength=0.3)
    
    # Show template if enabled
    if show_template and object_template is not None:
        template_display = cv2.resize(object_template, (100, 100))
        display_frame[10:110, frame.shape[1]-110:frame.shape[1]-10] = template_display
        cv2.rectangle(display_frame, (frame.shape[1]-110, 10), 
                     (frame.shape[1]-10, 110), (255, 255, 255), 2)
    
    cv2.putText(display_frame, "Q=Quit | SPACE=Pause | D=Mask | R=Reset | T=Template", 
               (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # Show mask overlay if enabled
    if show_mask:
        mask = detect_white_objects(frame)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display_frame = cv2.addWeighted(display_frame, 0.7, mask_colored, 0.3, 0)
    
    cv2.imshow("Robust White Car Tracking", display_frame)
    
    # Terminal output with detailed info
    if frame_count % 20 == 0:
        status_str = 'ACTIVE' if tracking_active else ('INIT' if not tracking_initialized else 'LOST')
        current_rate = (total_detections/frame_count)*100 if frame_count > 0 else 0
        print(f"F{frame_count}: Det={'✓' if detection_found else '✗'} | "
              f"Track={status_str} | Lost={lost_frames} | Rate={current_rate:.1f}%")
    
    # SANITY CHECK - Detection rate should NEVER exceed 100%
    if total_detections > frame_count:
        print(f"\n⚠️ WARNING: Bug detected! total_detections ({total_detections}) > frame_count ({frame_count})")
        print(f"   This should be impossible. Detections this frame: {detections_this_frame}")
        total_detections = frame_count  # Fix it
    
    # KEY HANDLING
    key = cv2.waitKey(30) & 0xFF
    
    if key == 27 or key == ord('q'):
        print("\n👋 Quitting...")
        break
    elif key == ord(' '):
        print("\n⏸  PAUSED")
        cv2.waitKey(0)
        print("▶  RESUMED")
    elif key == ord('d'):
        show_mask = not show_mask
    elif key == ord('t'):
        show_template = not show_template
    elif key == ord('r'):
        print("\n🔄 RESETTING TRACKING")
        tracking_initialized = False
        tracking_active = False
        kf = None
        object_template = None
        lost_frames = 0

cap.release()
cv2.destroyAllWindows()

# FINAL STATISTICS
print("\n" + "=" * 70)
print("FINAL STATISTICS")
print("=" * 70)
print(f"Total Frames: {frame_count}")
print(f"Detections: {total_detections} ({(total_detections/frame_count)*100:.1f}%)")
print(f"Template Matches: {template_matches}")
print(f"Recovery Attempts: {recovery_attempts}")
print(f"Successful Recoveries: {successful_recoveries}")
if recovery_attempts > 0:
    print(f"Recovery Rate: {(successful_recoveries/recovery_attempts)*100:.1f}%")
print(f"Out-of-Frame Warnings: {out_of_frame_warnings}")
print(f"Final Status: {'ACTIVE' if tracking_active else 'LOST'}")
print("=" * 70)