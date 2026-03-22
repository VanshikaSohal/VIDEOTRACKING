import cv2
import numpy as np
from collections import deque

# ========================
# KALMAN TRACKER CLASS
# ========================
class KalmanTracker:
    def __init__(self, x, y, tracker_id):
        self.id = tracker_id
        self.lost_frames = 0
        self.active = True
        self.trail = deque(maxlen=40)

        self.kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
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
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 6.0
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)

    def predict(self):
        pred = self.kf.predict()
        px, py = int(pred[0][0]), int(pred[1][0])
        self.trail.append((px, py))
        return px, py

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)


# =======================
# DETECTION FUNCTIONS
# =======================
def detect_motion(frame, prev_frame):
    if prev_frame is None or frame.shape != prev_frame.shape:
        return []
    g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, None, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 1200]


def detect_bg(frame, bg):
    fg = bg.apply(frame)
    fg = cv2.medianBlur(fg, 5)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 1500]


def detect_hog(frame, hog):
    rects, _ = hog.detectMultiScale(frame, winStride=(8, 8))
    return list(rects)


def merge_detections(dets, min_dist=50):
    """Merge nearby bounding boxes to avoid duplicate detections."""
    merged = []
    used = [False] * len(dets)
    for i, d1 in enumerate(dets):
        if used[i]:
            continue
        cx1 = d1[0] + d1[2] // 2
        cy1 = d1[1] + d1[3] // 2
        group = [d1]
        for j, d2 in enumerate(dets):
            if i == j or used[j]:
                continue
            cx2 = d2[0] + d2[2] // 2
            cy2 = d2[1] + d2[3] // 2
            if np.hypot(cx1 - cx2, cy1 - cy2) < min_dist:
                group.append(d2)
                used[j] = True
        # Take the largest box in the group
        merged.append(max(group, key=lambda b: b[2] * b[3]))
        used[i] = True
    return merged


def get_top_n_detections(dets, n=2):
    """Return top-N largest detections after merging."""
    merged = merge_detections(dets)
    merged.sort(key=lambda b: b[2] * b[3], reverse=True)
    return merged[:n]


# =====================
# ASSIGNMENT FUNCTION
# =====================
def assign_detections_to_trackers(trackers, detections, predictions, max_dist=300):
    """
    Greedy nearest-distance assignment.
    Returns: list of (tracker_idx, detection_box) pairs
    """
    assignments = []
    assigned_dets = set()

    for t_idx, (pred_x, pred_y) in enumerate(predictions):
        best_dist = max_dist
        best_d_idx = -1
        for d_idx, (x, y, w, h) in enumerate(detections):
            if d_idx in assigned_dets:
                continue
            cx, cy = x + w // 2, y + h // 2
            d = np.hypot(cx - pred_x, cy - pred_y)
            if d < best_dist:
                best_dist = d
                best_d_idx = d_idx

        if best_d_idx >= 0:
            assignments.append((t_idx, detections[best_d_idx]))
            assigned_dets.add(best_d_idx)

    return assignments


# ==================
# COLORS PER TRACKER
# ==================
TRACKER_COLORS = [
    (0, 0, 255),    # Red   → Object 1 prediction box
    (255, 128, 0),  # Orange → Object 2 prediction box
]
DETECTION_COLORS = [
    (0, 255, 0),    # Green → Object 1 detection
    (0, 255, 255),  # Yellow → Object 2 detection
]

MAX_OBJECTS = 2
MAX_LOST    = 90

cap = cv2.VideoCapture("twoobjectscurved.mp4")
ret, frame = cap.read()
if not ret:
    print("Cannot open video.")
    exit()

prev_frame = frame.copy()
H, W = frame.shape[:2]

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25)

trackers = []          # list of active KalmanTracker objects
next_id  = 0

total_frames          = 0
frames_with_detection = [0] * MAX_OBJECTS
accurate_detections   = [0] * MAX_OBJECTS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    # ── Step 1: Collect all detections ──
    raw_dets = detect_motion(frame, prev_frame) + detect_bg(frame, bg) + detect_hog(frame, hog)
    top_dets = get_top_n_detections(raw_dets, n=MAX_OBJECTS)

    # ── Step 2: If we have fewer trackers than objects, initialise new ones ──
    if len(trackers) < MAX_OBJECTS and top_dets:
        existing_centers = []
        for t in trackers:
            px, py = int(t.kf.statePost[0][0]), int(t.kf.statePost[1][0])
            existing_centers.append((px, py))

        for (x, y, w, h) in top_dets:
            if len(trackers) >= MAX_OBJECTS:
                break
            cx, cy = x + w // 2, y + h // 2
            # Don't init a tracker too close to an existing one
            too_close = any(np.hypot(cx - ex, cy - ey) < 80
                            for ex, ey in existing_centers)
            if not too_close:
                trackers.append(KalmanTracker(cx, cy, next_id))
                existing_centers.append((cx, cy))
                next_id += 1

    # ── Step 3: Predict all trackers ──
    predictions = [t.predict() for t in trackers]

    # ── Step 4: Assign detections to trackers ──
    # Use ROI-clipped detections per tracker for accuracy
    roi_dets_per_tracker = []
    for t_idx, t in enumerate(trackers):
        pred_x, pred_y = predictions[t_idx]
        ROI = min(250 + t.lost_frames * 20, 600)
        x1 = max(0, pred_x - ROI); y1 = max(0, pred_y - ROI)
        x2 = min(W, pred_x + ROI); y2 = min(H, pred_y + ROI)

        roi      = frame[y1:y2, x1:x2]
        prev_roi = prev_frame[y1:y2, x1:x2]

        local_dets = detect_motion(roi, prev_roi) + detect_bg(roi, bg) + detect_hog(roi, hog)
        # Convert local coords → global
        global_dets = [(x1+x, y1+y, w, h) for (x, y, w, h) in local_dets]
        roi_dets_per_tracker.append(global_dets)

    # Flatten all ROI detections and assign
    all_roi_dets = []
    for gdets in roi_dets_per_tracker:
        all_roi_dets.extend(gdets)
    all_roi_dets = merge_detections(all_roi_dets)

    assignments = assign_detections_to_trackers(trackers, all_roi_dets, predictions)

    assigned_tracker_ids = set()
    for (t_idx, bbox) in assignments:
        t = trackers[t_idx]
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        pred_x, pred_y = predictions[t_idx]
        dist = np.hypot(cx - pred_x, cy - pred_y)

        t.update(cx, cy)
        t.lost_frames = 0
        assigned_tracker_ids.add(t_idx)

        obj_idx = t_idx % MAX_OBJECTS
        frames_with_detection[obj_idx] += 1
        if dist < 40:
            accurate_detections[obj_idx] += 1

        # Green detection box
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      DETECTION_COLORS[obj_idx], 2)

    # ── Step 5: Increment lost count for unmatched trackers ──
    for t_idx, t in enumerate(trackers):
        if t_idx not in assigned_tracker_ids:
            t.lost_frames += 1

    # ── Step 6: Draw prediction boxes + trails ──
    for t_idx, t in enumerate(trackers):
        pred_x, pred_y = predictions[t_idx]
        obj_idx = t_idx % MAX_OBJECTS
        color = TRACKER_COLORS[obj_idx]

        scale = pred_y / H
        bw = int(60 + scale * 120)
        bh = int(120 + scale * 220)

        cv2.rectangle(frame,
                      (pred_x - bw//2, pred_y - bh//2),
                      (pred_x + bw//2, pred_y + bh//2),
                      color, 2)

        cv2.putText(frame, f"Obj {t.id}",
                    (pred_x - bw//2, pred_y - bh//2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw motion trail
        pts = list(t.trail)
        for i in range(1, len(pts)):
            if pts[i-1] and pts[i]:
                cv2.line(frame, pts[i-1], pts[i], color, 2)

    # ── Step 7: Remove dead trackers ──
    trackers = [t for t in trackers if t.lost_frames <= MAX_LOST]

    prev_frame = frame.copy()
    cv2.imshow("Dual Kalman Tracking", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ── METRICS ──
print(f"Total Frames: {total_frames}")
for i in range(MAX_OBJECTS):
    det_rate = (frames_with_detection[i] / total_frames) * 100
    print(f"\nObject {i+1}:")
    print(f"  Detection Rate   : {det_rate:.2f}%")
    if frames_with_detection[i] > 0:
        acc = (accurate_detections[i] / frames_with_detection[i]) * 100
        print(f"  Detection Accuracy: {acc:.2f}%")