

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
        self.id          = tracker_id
        self.lost_frames = 0
        self.trail       = deque(maxlen=40)
        self.dt          = dt

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
        self.ekf.R  = np.eye(2) * 5.0
        self.ekf.Q  = np.eye(4) * 0.05

        self.prev_positions = [(x, y)] * 3

    def predict(self):
        self.ekf.predict()
        px, py = int(self.ekf.x[0]), int(self.ekf.x[1])
        self.prev_positions.append((px, py))
        if len(self.prev_positions) > 3:
            self.prev_positions.pop(0)
        sx = int(np.mean([p[0] for p in self.prev_positions]))
        sy = int(np.mean([p[1] for p in self.prev_positions]))
        self.trail.append((sx, sy))
        return sx, sy

    def update(self, x, y):
        self.ekf.update(np.array([x, y], dtype=float),
                        HJacobian=H_jacobian, Hx=hx)

    def get_velocity(self):
        return self.ekf.x[2], self.ekf.x[3]

    def center(self):
        return int(self.ekf.x[0]), int(self.ekf.x[1])

def detect_motion(frame, prev_frame):
    if prev_frame is None or frame.shape != prev_frame.shape:
        return []
    g1 = cv2.GaussianBlur(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (21,21), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY), (21,21), 0)
    diff = cv2.absdiff(g1, g2)
    _, th = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    th = cv2.dilate(th, kernel, iterations=5)
    th = cv2.erode(th, kernel, iterations=3)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 400 < area < 50000:
            x, y, w, h = cv2.boundingRect(c)
            ar = h / float(w) if w > 0 else 0
            if 1.1 < ar < 5.5 and h > 18 and w > 12:
                pw, ph = int(w*0.6), int(h*0.4)
                boxes.append((max(0,x-pw), max(0,y-ph), w+2*pw, h+2*ph))
    return boxes

def detect_bg(frame, bg):
    fg = bg.apply(frame, learningRate=0.012)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 500 < area < 45000:
            x, y, w, h = cv2.boundingRect(c)
            ar = h / float(w) if w > 0 else 0
            if 1.1 < ar < 5.5 and h > 22 and w > 12:
                pw, ph = int(w*0.6), int(h*0.4)
                boxes.append((max(0,x-pw), max(0,y-ph), w+2*pw, h+2*ph))
    return boxes

def combine_detections(boxes, max_dist=180):
    if not boxes:
        return []
    merged, used = [], set()
    for i, (x1,y1,w1,h1) in enumerate(boxes):
        if i in used: continue
        cx1, cy1 = x1+w1//2, y1+h1//2
        group = [(x1,y1,w1,h1)]
        for j, (x2,y2,w2,h2) in enumerate(boxes[i+1:], i+1):
            if j in used: continue
            if np.hypot(cx1-(x2+w2//2), cy1-(y2+h2//2)) < max_dist:
                group.append((x2,y2,w2,h2)); used.add(j)
        ax  = [b[0] for b in group]
        ay  = [b[1] for b in group]
        ax2 = [b[0]+b[2] for b in group]
        ay2 = [b[1]+b[3] for b in group]
        merged.append((min(ax), min(ay), max(ax2)-min(ax), max(ay2)-min(ay)))
        used.add(i)
    return merged

def assign_detections_to_trackers(trackers, detections, max_dist=300):
    assignments, assigned = [], set()
    for t_idx, t in enumerate(trackers):
        tcx, tcy = t.center()
        best_dist, best_d = max_dist, -1
        for d_idx, (x,y,w,h) in enumerate(detections):
            if d_idx in assigned: continue
            d = np.hypot(x+w//2-tcx, y+h//2-tcy)
            if d < best_dist:
                best_dist, best_d = d, d_idx
        if best_d >= 0:
            assignments.append((t_idx, detections[best_d])); assigned.add(best_d)
    return assignments


# Colors: [Obj1, Obj2]
PRED_COLORS = [(0, 0, 255),   (255, 128, 0)]
DET_COLORS  = [(0, 255, 0),   (0, 255, 255)]
MAX_OBJECTS = 2
MAX_LOST    = 50
MIN_CONF    = 3   
print("="*60)
print("EKF DUAL TRACKER - NO HOG - TWO OBJECTS")
print("="*60)

VIDEO_FILE = "twoobjectscurved.mp4"
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f" Cannot open '{VIDEO_FILE}'"); exit()

ret, frame = cap.read()
if not ret:
    print(" Cannot read first frame"); exit()

prev_frame = frame.copy()
H, W = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"📹 {W}x{H} @ {fps:.1f} FPS\n🎮 ESC=Quit | SPACE=Pause | R=Reset\n")

bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=18, detectShadows=False)

trackers        = []
next_id         = 0
consecutive     = 0  

total_frames          = 0
frames_with_detection = [0] * MAX_OBJECTS
accurate_detections   = [0] * MAX_OBJECTS

bbox_histories = [[] for _ in range(MAX_OBJECTS)]
MAX_BH = 4

while True:
    ret, frame = cap.read()
    if not ret: break
    total_frames += 1

    all_dets = combine_detections(
        detect_motion(frame, prev_frame) + detect_bg(frame, bg), max_dist=200)
    all_dets = [b for b in all_dets if b[2] > 20 and b[3] > 45]

    if len(trackers) < MAX_OBJECTS and all_dets:
        consecutive += 1
        if consecutive >= MIN_CONF:
            existing = [t.center() for t in trackers]
            all_dets_sorted = sorted(all_dets, key=lambda b: b[2]*b[3], reverse=True)
            for (x, y, w, h) in all_dets_sorted:
                if len(trackers) >= MAX_OBJECTS: break
                cx, cy = x+w//2, y+h//2
                if not any(np.hypot(cx-ex, cy-ey) < 80 for ex, ey in existing):
                    trackers.append(EKFTracker(cx, cy, next_id,
                                               dt=1.0/fps if fps > 0 else 0.033))
                    existing.append((cx, cy))
                    print(f"✅ Frame {total_frames}: EKF Obj {next_id} initialised")
                    next_id += 1
    else:
        if not all_dets: consecutive = 0

    if not trackers:
        prev_frame = frame.copy()
        cv2.imshow("EKF Dual Tracking - No HOG", frame)
        if cv2.waitKey(30) & 0xFF == 27: break
        continue

    predictions = [t.predict() for t in trackers]

    all_roi_dets = []
    for t_idx, (pred_x, pred_y) in enumerate(predictions):
        t = trackers[t_idx]
        ROI = min(280 + t.lost_frames * 25, 650)
        x1 = max(0, pred_x-ROI); y1 = max(0, pred_y-ROI)
        x2 = min(W, pred_x+ROI); y2 = min(H, pred_y+ROI)
        roi      = frame[y1:y2, x1:x2]
        prev_roi = prev_frame[y1:y2, x1:x2]
        local = combine_detections(
            detect_motion(roi, prev_roi) + detect_bg(roi, bg), max_dist=200)
        all_roi_dets.extend([(x1+x, y1+y, w, h) for (x,y,w,h) in local])

    all_roi_dets = combine_detections(all_roi_dets)
    assignments  = assign_detections_to_trackers(trackers, all_roi_dets)
    assigned_ids = set()

    for (t_idx, (x, y, w, h)) in assignments:
        t = trackers[t_idx]
        pred_x, pred_y = predictions[t_idx]
        cx, cy = x+w//2, y+h//2
        dist = np.hypot(cx-pred_x, cy-pred_y)
        dist_threshold = 0.65 * min(280 + t.lost_frames*25, 650)

        if dist < dist_threshold:
            t.update(cx, cy)
            t.lost_frames = 0
            assigned_ids.add(t_idx)

            obj_idx = t_idx % MAX_OBJECTS
            frames_with_detection[obj_idx] += 1
            if dist < 45: accurate_detections[obj_idx] += 1

            bbox_histories[obj_idx].append((x, y, w, h))
            if len(bbox_histories[obj_idx]) > MAX_BH:
                bbox_histories[obj_idx].pop(0)

            # Detection box
            cv2.rectangle(frame, (x,y), (x+w,y+h), DET_COLORS[obj_idx], 3)
            cv2.putText(frame, f"EKF Obj {t.id} DETECT",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DET_COLORS[obj_idx], 2)

    for t_idx, t in enumerate(trackers):
        pred_x, pred_y = predictions[t_idx]
        obj_idx = t_idx % MAX_OBJECTS
        color = PRED_COLORS[obj_idx]

        if t_idx not in assigned_ids:
            t.lost_frames += 1

        # Velocity-adaptive prediction box
        vx, vy = t.get_velocity()
        vel = np.sqrt(vx**2 + vy**2)
        scale = pred_y / H
        bw = max(50, min(int(65 + scale*110 + vel*5), 200))
        bh = max(100, min(int(130 + scale*200 + vel*8), 350))

        cv2.rectangle(frame,
                      (pred_x-bw//2, pred_y-bh//2),
                      (pred_x+bw//2, pred_y+bh//2), color, 3)
        label = "DETECT" if t_idx in assigned_ids else f"PRED ({t.lost_frames}f)"
        cv2.putText(frame, f"EKF Obj {t.id} {label}",
                    (pred_x-bw//2, pred_y-bh//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Trail
        pts = list(t.trail)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], color, 2)

    before   = len(trackers)
    trackers = [t for t in trackers if t.lost_frames <= MAX_LOST]
    if len(trackers) < before:
        print(f" Frame {total_frames}: Lost a tracker")
        consecutive = 0

    cv2.rectangle(frame, (0,0), (W, 70), (40,40,40), -1)
    cv2.putText(frame, f"Frame: {total_frames} | EKF Tracking {len(trackers)}/{MAX_OBJECTS}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Det: Obj1={frames_with_detection[0]} Obj2={frames_with_detection[1]}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    prev_frame = frame.copy()
    cv2.imshow("EKF Dual Tracking - No HOG", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27: break
    elif key == ord(' '): cv2.waitKey(0)
    elif key == ord('r'):
        trackers = []; consecutive = 0; next_id = 0
        bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=18, detectShadows=False)
        print("🔄 RESET")

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*60}\nFINAL STATISTICS\n{'='*60}")
print(f"Total Frames: {total_frames}")
for i in range(MAX_OBJECTS):
    print(f"\nObject {i+1}:")
    print(f"  Detection Rate:     {(frames_with_detection[i]/total_frames)*100:.1f}%")
    if frames_with_detection[i] > 0:
        print(f"  Detection Accuracy: {(accurate_detections[i]/frames_with_detection[i])*100:.1f}%")