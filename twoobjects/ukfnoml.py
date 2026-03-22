"""
3D UKF PERSON TRACKER - NO HOG - TWO OBJECTS
"""

import cv2
import numpy as np
from collections import deque
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# =======================
# CAMERA MODEL & PRIORS
# =======================
class CameraModel:
    def __init__(self, img_width, img_height, focal_length_px=None):
        self.img_width = img_width
        self.img_height = img_height
        self.focal_length = focal_length_px if focal_length_px else img_width * 1.2
        self.cx = img_width / 2.0
        self.cy = img_height / 2.0

    def estimate_depth_from_height(self, bbox_height_px, real_height_m=1.7):
        if bbox_height_px < 10: return None
        return (self.focal_length * real_height_m) / bbox_height_px

    def project_3d_to_2d(self, X, Y, Z):
        Z = max(Z, 0.1)
        u = self.focal_length * (X / Z) + self.cx
        v = self.focal_length * (Y / Z) + self.cy
        return u, v

class PedestrianPriors:
    def __init__(self):
        self.height_mean = 1.70
        self.height_std  = 0.15
        self.width_mean  = 0.50

# =======================
# 3D UKF TRACKER
# =======================
def fx_3d(x, dt):
    F = np.array([
        [1,0,0,dt,0,0],
        [0,1,0,0,dt,0],
        [0,0,1,0,0,dt],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]
    ], dtype=float)
    return F @ x

def hx_3d(x, camera):
    X, Y, Z = x[0], x[1], x[2]
    u, v = camera.project_3d_to_2d(X, Y, Z)
    bbox_h = (camera.focal_length * 1.7) / max(Z, 0.1)
    return np.array([u, v, bbox_h])

class UKFTracker3D:
    def __init__(self, u, v, bbox_height, tracker_id, camera, priors, dt=1.0):
        self.id          = tracker_id
        self.lost_frames = 0
        self.trail       = deque(maxlen=40)
        self.camera      = camera
        self.priors      = priors

        depth = camera.estimate_depth_from_height(bbox_height, priors.height_mean)
        if depth is None or not (0.5 < depth < 50): depth = 5.0

        X_init = (u - camera.cx) * depth / camera.focal_length
        Z_init = depth

        points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=1)
        self.ukf = UKF(dim_x=6, dim_z=3,
                       fx=lambda x, dt: fx_3d(x, dt),
                       hx=lambda x: hx_3d(x, camera),
                       dt=dt, points=points)

        self.ukf.x  = np.array([X_init, 0., Z_init, 0., 0., 0.], dtype=float)
        self.ukf.P *= 5.0
        self.ukf.R  = np.diag([10.0, 10.0, 20.0])
        self.ukf.Q  = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])

        self.depth_history = [Z_init] * 3

    def predict(self):
        self.ukf.predict()
        X, Y, Z = self.ukf.x[0], self.ukf.x[1], max(self.ukf.x[2], 0.5)
        u, v = self.camera.project_3d_to_2d(X, Y, Z)
        self.depth_history.append(Z)
        if len(self.depth_history) > 5: self.depth_history.pop(0)
        smooth_depth = np.median(self.depth_history)
        pu, pv = int(u), int(v)
        self.trail.append((pu, pv))
        return pu, pv, smooth_depth

    def update(self, u, v, bbox_height):
        self.ukf.update(np.array([u, v, bbox_height], dtype=float))
        self.ukf.x[2] = np.clip(self.ukf.x[2], 0.5, 50)

    def get_3d_state(self):
        return self.ukf.x[0], self.ukf.x[1], self.ukf.x[2], \
               self.ukf.x[3], self.ukf.x[4], self.ukf.x[5]

    def get_predicted_bbox_size(self):
        Z = max(self.ukf.x[2], 0.1)
        ph = int((self.camera.focal_length * self.priors.height_mean) / Z)
        pw = int((self.camera.focal_length * self.priors.width_mean)  / Z)
        return pw, ph

    def center(self):
        X, Y, Z = self.ukf.x[0], self.ukf.x[1], max(self.ukf.x[2], 0.5)
        u, v = self.camera.project_3d_to_2d(X, Y, Z)
        return int(u), int(v)


# =======================
# DETECTION FUNCTIONS
# =======================
def detect_motion(frame, prev_frame):
    if prev_frame is None or frame.shape != prev_frame.shape: return []
    g1 = cv2.GaussianBlur(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),(21,21),0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY),(21,21),0)
    diff = cv2.absdiff(g1, g2)
    _, th = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17,17))
    th = cv2.dilate(th, kernel, iterations=5)
    th = cv2.erode(th, kernel, iterations=3)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 400 < area < 50000:
            x, y, w, h = cv2.boundingRect(c)
            ar = h/float(w) if w > 0 else 0
            if 1.1 < ar < 5.5 and h > 18 and w > 12:
                pw, ph = int(w*0.6), int(h*0.4)
                boxes.append((max(0,x-pw), max(0,y-ph), w+2*pw, h+2*ph))
    return boxes

def detect_bg(frame, bg):
    fg = bg.apply(frame, learningRate=0.012)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 500 < area < 45000:
            x, y, w, h = cv2.boundingRect(c)
            ar = h/float(w) if w > 0 else 0
            if 1.1 < ar < 5.5 and h > 22 and w > 12:
                pw, ph = int(w*0.6), int(h*0.4)
                boxes.append((max(0,x-pw), max(0,y-ph), w+2*pw, h+2*ph))
    return boxes

def combine_detections(boxes, max_dist=180):
    if not boxes: return []
    merged, used = [], set()
    for i, (x1,y1,w1,h1) in enumerate(boxes):
        if i in used: continue
        cx1, cy1 = x1+w1//2, y1+h1//2
        group = [(x1,y1,w1,h1)]
        for j, (x2,y2,w2,h2) in enumerate(boxes[i+1:], i+1):
            if j in used: continue
            if np.hypot(cx1-(x2+w2//2), cy1-(y2+h2//2)) < max_dist:
                group.append((x2,y2,w2,h2)); used.add(j)
        ax  = [b[0] for b in group]; ay  = [b[1] for b in group]
        ax2 = [b[0]+b[2] for b in group]; ay2 = [b[1]+b[3] for b in group]
        merged.append((min(ax),min(ay),max(ax2)-min(ax),max(ay2)-min(ay)))
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
            if d < best_dist: best_dist, best_d = d, d_idx
        if best_d >= 0:
            assignments.append((t_idx, detections[best_d])); assigned.add(best_d)
    return assignments


# Colors: [Obj1, Obj2]
PRED_COLORS = [(0, 0, 255),  (255, 128, 0)]
DET_COLORS  = [(0, 255, 0),  (0, 255, 255)]
MAX_OBJECTS = 2
MAX_LOST    = 50
MIN_CONF    = 3


# =======================
# MAIN
# =======================
print("="*60)
print("3D UKF DUAL TRACKER - NO HOG - TWO OBJECTS")
print("="*60)

VIDEO_FILE = "twoobjectscurved.mp4"
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"❌ Cannot open '{VIDEO_FILE}'"); exit()

ret, frame = cap.read()
if not ret:
    print("❌ Cannot read first frame"); exit()

prev_frame = frame.copy()
H, W = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"📹 {W}x{H} @ {fps:.1f} FPS\n🎮 ESC=Quit | SPACE=Pause | R=Reset\n")

camera  = CameraModel(W, H)
priors  = PedestrianPriors()
bg      = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=18, detectShadows=False)

trackers    = []
next_id     = 0
consecutive = 0

total_frames          = 0
frames_with_detection = [0] * MAX_OBJECTS
accurate_detections   = [0] * MAX_OBJECTS
bbox_histories        = [[] for _ in range(MAX_OBJECTS)]
MAX_BH = 4

while True:
    ret, frame = cap.read()
    if not ret: break
    total_frames += 1

    # ── Full-frame detections ──
    all_dets = combine_detections(
        detect_motion(frame, prev_frame) + detect_bg(frame, bg), max_dist=200)
    all_dets = [b for b in all_dets if b[2] > 20 and b[3] > 45]

    # ── Init new trackers ──
    if len(trackers) < MAX_OBJECTS and all_dets:
        consecutive += 1
        if consecutive >= MIN_CONF:
            existing = [t.center() for t in trackers]
            for (x, y, w, h) in sorted(all_dets, key=lambda b: b[2]*b[3], reverse=True):
                if len(trackers) >= MAX_OBJECTS: break
                cx, cy = x+w//2, y+h//2
                if not any(np.hypot(cx-ex, cy-ey) < 80 for ex, ey in existing):
                    trackers.append(UKFTracker3D(cx, cy, h, next_id, camera, priors,
                                                 dt=1.0/fps if fps > 0 else 0.033))
                    existing.append((cx, cy))
                    print(f"✅ Frame {total_frames}: 3D UKF Obj {next_id} initialised")
                    next_id += 1
    else:
        if not all_dets: consecutive = 0

    if not trackers:
        prev_frame = frame.copy()
        cv2.imshow("3D UKF Dual Tracking - No HOG", frame)
        if cv2.waitKey(30) & 0xFF == 27: break
        continue

    # ── Predict ──
    predictions = [t.predict() for t in trackers]   # (pred_u, pred_v, depth)

    # ── ROI detections ──
    all_roi_dets = []
    for t_idx, (pu, pv, _) in enumerate(predictions):
        t = trackers[t_idx]
        ROI = min(280 + t.lost_frames*25, 650)
        x1=max(0,pu-ROI); y1=max(0,pv-ROI)
        x2=min(W,pu+ROI); y2=min(H,pv+ROI)
        roi=frame[y1:y2,x1:x2]; prev_roi=prev_frame[y1:y2,x1:x2]
        local = combine_detections(detect_motion(roi,prev_roi)+detect_bg(roi,bg), max_dist=200)
        all_roi_dets.extend([(x1+x,y1+y,w,h) for (x,y,w,h) in local])

    all_roi_dets = combine_detections(all_roi_dets)
    assignments  = assign_detections_to_trackers(trackers, all_roi_dets)
    assigned_ids = set()

    for (t_idx, (x, y, w, h)) in assignments:
        t = trackers[t_idx]
        pu, pv, _ = predictions[t_idx]
        cx, cy = x+w//2, y+h//2
        dist = np.hypot(cx-pu, cy-pv)
        dist_threshold = 0.65 * min(280+t.lost_frames*25, 650)

        if dist < dist_threshold:
            t.update(cx, cy, h)
            t.lost_frames = 0
            assigned_ids.add(t_idx)

            obj_idx = t_idx % MAX_OBJECTS
            frames_with_detection[obj_idx] += 1
            if dist < 45: accurate_detections[obj_idx] += 1

            bbox_histories[obj_idx].append((x,y,w,h))
            if len(bbox_histories[obj_idx]) > MAX_BH:
                bbox_histories[obj_idx].pop(0)

            cv2.rectangle(frame, (x,y), (x+w,y+h), DET_COLORS[obj_idx], 3)
            cv2.putText(frame, f"UKF Obj {t.id} DETECT (h={h}px)",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DET_COLORS[obj_idx], 2)

    # ── Unmatched trackers ──
    for t_idx, t in enumerate(trackers):
        pu, pv, depth = predictions[t_idx]
        obj_idx = t_idx % MAX_OBJECTS
        color = PRED_COLORS[obj_idx]

        if t_idx not in assigned_ids:
            t.lost_frames += 1

        bw, bh = t.get_predicted_bbox_size()
        bw = max(40, min(bw, 200)); bh = max(80, min(bh, 350))

        cv2.rectangle(frame,
                      (pu-bw//2, pv-bh//2),
                      (pu+bw//2, pv+bh//2), color, 3)
        label = "DETECT" if t_idx in assigned_ids else f"PRED ({t.lost_frames}f)"
        cv2.putText(frame, f"UKF Obj {t.id} {label} Z={depth:.1f}m",
                    (pu-bw//2, pv-bh//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 3D state HUD per object
        X, Y, Z, vX, vY, vZ = t.get_3d_state()
        vel = np.sqrt(vX**2+vY**2+vZ**2)
        cv2.putText(frame, f"v={vel:.2f}m/s",
                    (pu-bw//2, pv+bh//2+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Trail
        pts = list(t.trail)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], color, 2)

    # ── Remove dead trackers ──
    before   = len(trackers)
    trackers = [t for t in trackers if t.lost_frames <= MAX_LOST]
    if len(trackers) < before:
        print(f"⚠️ Frame {total_frames}: Lost a tracker"); consecutive = 0

    # ── HUD ──
    cv2.rectangle(frame,(0,0),(W,100),(40,40,40),-1)
    cv2.putText(frame, f"Frame: {total_frames} | 3D UKF Tracking {len(trackers)}/{MAX_OBJECTS}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if trackers:
        X,Y,Z,vX,vY,vZ = trackers[0].get_3d_state()
        cv2.putText(frame, f"Obj0: X={X:.1f} Y={Y:.1f} Z={Z:.1f}m",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, PRED_COLORS[0], 2)
    if len(trackers) > 1:
        X,Y,Z,vX,vY,vZ = trackers[1].get_3d_state()
        cv2.putText(frame, f"Obj1: X={X:.1f} Y={Y:.1f} Z={Z:.1f}m",
                    (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, PRED_COLORS[1], 2)

    prev_frame = frame.copy()
    cv2.imshow("3D UKF Dual Tracking - No HOG", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27: break
    elif key == ord(' '): cv2.waitKey(0)
    elif key == ord('r'):
        trackers=[]; consecutive=0; next_id=0
        bg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=18,detectShadows=False)
        print("🔄 RESET")

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*60}\nFINAL STATISTICS - 3D DUAL TRACKING\n{'='*60}")
print(f"Total Frames: {total_frames}")
for i in range(MAX_OBJECTS):
    print(f"\nObject {i+1}:")
    print(f"  Detection Rate:     {(frames_with_detection[i]/total_frames)*100:.1f}%")
    if frames_with_detection[i] > 0:
        print(f"  Detection Accuracy: {(accurate_detections[i]/frames_with_detection[i])*100:.1f}%")