import cv2
import numpy as np
from collections import deque
from filterpy.kalman import ExtendedKalmanFilter as EKF

def hx(x):
    return np.array([x[0], x[1]])

def H_jacobian(x):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

class EKFTracker:
    def __init__(self, x, y, tracker_id, dt=1.0):
        self.id = tracker_id
        self.lost_frames = 0
        self.trail = deque(maxlen=40)
        self.dt = dt

        self.ekf = EKF(dim_x=4, dim_z=2)
        self.ekf.x = np.array([x, y, 0, 0], dtype=float)
        self.ekf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)
        self.ekf.H  = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], dtype=float)
        self.ekf.P *= 10.0
        self.ekf.R  = np.eye(2) * 6.0
        self.ekf.Q  = np.eye(4) * 0.03

    def predict(self):
        self.ekf.predict()
        px, py = int(self.ekf.x[0]), int(self.ekf.x[1])
        self.trail.append((px, py))
        return px, py

    def update(self, x, y):
        z = np.array([x, y])
        self.ekf.update(z, HJacobian=H_jacobian, Hx=hx)


def detect_motion(frame, prev_frame):
    if prev_frame is None or frame.shape != prev_frame.shape:
        return []
    g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY)
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
    merged = []
    used = [False] * len(dets)
    for i, d1 in enumerate(dets):
        if used[i]:
            continue
        cx1, cy1 = d1[0] + d1[2]//2, d1[1] + d1[3]//2
        group = [d1]
        for j, d2 in enumerate(dets):
            if i == j or used[j]:
                continue
            cx2, cy2 = d2[0] + d2[2]//2, d2[1] + d2[3]//2
            if np.hypot(cx1-cx2, cy1-cy2) < min_dist:
                group.append(d2)
                used[j] = True
        merged.append(max(group, key=lambda b: b[2]*b[3]))
        used[i] = True
    return merged

def get_top_n_detections(dets, n=2):
    merged = merge_detections(dets)
    merged.sort(key=lambda b: b[2]*b[3], reverse=True)
    return merged[:n]

def assign_detections_to_trackers(predictions, detections, max_dist=300):
    assignments = []
    assigned_dets = set()
    for t_idx, (pred_x, pred_y) in enumerate(predictions):
        best_dist, best_d_idx = max_dist, -1
        for d_idx, (x, y, w, h) in enumerate(detections):
            if d_idx in assigned_dets:
                continue
            d = np.hypot(x+w//2 - pred_x, y+h//2 - pred_y)
            if d < best_dist:
                best_dist, best_d_idx = d, d_idx
        if best_d_idx >= 0:
            assignments.append((t_idx, detections[best_d_idx]))
            assigned_dets.add(best_d_idx)
    return assignments


# Colors: [Obj1, Obj2]
PRED_COLORS = [(0, 0, 255), (255, 128, 0)]
DET_COLORS  = [(0, 255, 0), (0, 255, 255)]
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

trackers = []
next_id   = 0

total_frames          = 0
frames_with_detection = [0] * MAX_OBJECTS
accurate_detections   = [0] * MAX_OBJECTS

while True:
    ret, frame = cap.read()
    if not ret:
        break
    total_frames += 1

    raw_dets = detect_motion(frame, prev_frame) + detect_bg(frame, bg) + detect_hog(frame, hog)
    top_dets = get_top_n_detections(raw_dets, n=MAX_OBJECTS)

    if len(trackers) < MAX_OBJECTS and top_dets:
        existing = [(int(t.ekf.x[0]), int(t.ekf.x[1])) for t in trackers]
        for (x, y, w, h) in top_dets:
            if len(trackers) >= MAX_OBJECTS:
                break
            cx, cy = x+w//2, y+h//2
            if not any(np.hypot(cx-ex, cy-ey) < 80 for ex, ey in existing):
                trackers.append(EKFTracker(cx, cy, next_id))
                existing.append((cx, cy))
                next_id += 1

    predictions = [t.predict() for t in trackers]

    all_roi_dets = []
    for t_idx, t in enumerate(trackers):
        pred_x, pred_y = predictions[t_idx]
        ROI = min(250 + t.lost_frames * 20, 600)
        x1 = max(0, pred_x-ROI); y1 = max(0, pred_y-ROI)
        x2 = min(W, pred_x+ROI); y2 = min(H, pred_y+ROI)
        roi      = frame[y1:y2, x1:x2]
        prev_roi = prev_frame[y1:y2, x1:x2]
        local = detect_motion(roi, prev_roi) + detect_bg(roi, bg) + detect_hog(roi, hog)
        all_roi_dets.extend([(x1+x, y1+y, w, h) for (x, y, w, h) in local])

    all_roi_dets = merge_detections(all_roi_dets)
    assignments  = assign_detections_to_trackers(predictions, all_roi_dets)

    assigned_ids = set()
    for (t_idx, bbox) in assignments:
        t = trackers[t_idx]
        x, y, w, h = bbox
        cx, cy = x+w//2, y+h//2
        pred_x, pred_y = predictions[t_idx]
        dist = np.hypot(cx-pred_x, cy-pred_y)

        t.update(cx, cy)
        t.lost_frames = 0
        assigned_ids.add(t_idx)

        obj_idx = t_idx % MAX_OBJECTS
        frames_with_detection[obj_idx] += 1
        if dist < 40:
            accurate_detections[obj_idx] += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), DET_COLORS[obj_idx], 2)

    for t_idx, t in enumerate(trackers):
        if t_idx not in assigned_ids:
            t.lost_frames += 1

    for t_idx, t in enumerate(trackers):
        pred_x, pred_y = predictions[t_idx]
        obj_idx = t_idx % MAX_OBJECTS
        color = PRED_COLORS[obj_idx]
        scale = pred_y / H
        bw = int(60 + scale * 120)
        bh = int(120 + scale * 220)
        cv2.rectangle(frame,
                      (pred_x-bw//2, pred_y-bh//2),
                      (pred_x+bw//2, pred_y+bh//2), color, 2)
        cv2.putText(frame, f"EKF Obj {t.id}",
                    (pred_x-bw//2, pred_y-bh//2-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        pts = list(t.trail)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], color, 2)

    trackers = [t for t in trackers if t.lost_frames <= MAX_LOST]

    prev_frame = frame.copy()
    cv2.imshow("EKF Dual Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total Frames: {total_frames}")
for i in range(MAX_OBJECTS):
    print(f"\nObject {i+1}:")
    print(f"  Detection Rate:    {(frames_with_detection[i]/total_frames)*100:.2f}%")
    if frames_with_detection[i] > 0:
        print(f"  Detection Accuracy:{(accurate_detections[i]/frames_with_detection[i])*100:.2f}%")