"""
3D UKF DUAL TRACKER - OCCLUSION HANDLING - FIXED
"""

import cv2
import numpy as np
from collections import deque
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# =======================
# CAMERA MODEL
# =======================
class CameraModel:
    def __init__(self, W, H, focal_length_px=None):
        self.focal_length = focal_length_px if focal_length_px else W * 1.2
        self.cx = W / 2.0
        self.cy = H / 2.0
        self.W  = W
        self.H  = H

    def estimate_depth(self, bbox_h_px, real_h=1.7):
        if bbox_h_px < 10:
            return None
        return (self.focal_length * real_h) / bbox_h_px

    def project(self, X, Y, Z):
        Z = max(Z, 0.1)
        u = self.focal_length * (X / Z) + self.cx
        v = self.focal_length * (Y / Z) + self.cy
        return u, v

class PedestrianPriors:
    height_mean = 1.70
    width_mean  = 0.50


# =======================
# UKF FUNCTIONS
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
    X, Y, Z = x[0], x[1], max(x[2], 0.1)
    u, v = camera.project(X, Y, Z)
    bh = (camera.focal_length * 1.7) / Z
    return np.array([u, v, bh])


# =======================
# UKF TRACKER
# =======================
class UKFTracker3D:
    def __init__(self, u, v, bbox_h, tracker_id, camera, dt=1.0):
        self.id              = tracker_id
        self.lost_frames     = 0
        self.trail           = deque(maxlen=50)
        self.camera          = camera
        self.is_occluded     = False
        self.occluded_frames = 0
        self.last_known_pos  = (u, v)

        depth = camera.estimate_depth(bbox_h)
        if depth is None or not (0.5 < depth < 50):
            depth = 5.0

        X_init = (u - camera.cx) * depth / camera.focal_length
        Z_init = depth

        points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=1)
        self.ukf = UKF(
            dim_x=6, dim_z=3,
            fx=lambda x, dt: fx_3d(x, dt),
            hx=lambda x: hx_3d(x, camera),
            dt=dt, points=points
        )
        self.ukf.x  = np.array([X_init, 0., Z_init, 0., 0., 0.], dtype=float)
        self.ukf.P *= 5.0
        self.ukf.R  = np.diag([10., 10., 20.])
        self.ukf.Q  = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        self.depth_history = [Z_init] * 3

    def predict(self):
        if self.is_occluded:
            self.ukf.Q = np.diag([0.3, 0.3, 0.3, 1.0, 1.0, 1.0])
        else:
            self.ukf.Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])

        self.ukf.predict()
        X, Y, Z = self.ukf.x[0], self.ukf.x[1], max(self.ukf.x[2], 0.5)
        u, v    = self.camera.project(X, Y, Z)

        # Clamp projected position to frame bounds
        u = float(np.clip(u, 0, self.camera.W - 1))
        v = float(np.clip(v, 0, self.camera.H - 1))

        self.depth_history.append(Z)
        if len(self.depth_history) > 5:
            self.depth_history.pop(0)
        depth = float(np.median(self.depth_history))

        pu, pv = int(u), int(v)
        self.trail.append((pu, pv))
        return pu, pv, depth

    def update(self, u, v, bbox_h):
        self.ukf.update(np.array([float(u), float(v), float(bbox_h)]))
        self.ukf.x[2]       = float(np.clip(self.ukf.x[2], 0.5, 50))
        self.last_known_pos  = (u, v)
        self.is_occluded     = False
        self.occluded_frames = 0

    def get_predicted_bbox_size(self):
        Z  = max(self.ukf.x[2], 0.1)
        ph = int((self.camera.focal_length * 1.7) / Z)
        pw = int((self.camera.focal_length * 0.5) / Z)
        return pw, ph

    def get_3d_state(self):
        return self.ukf.x.copy()

    def center(self):
        X, Y, Z = self.ukf.x[0], self.ukf.x[1], max(self.ukf.x[2], 0.5)
        u, v    = self.camera.project(X, Y, Z)
        u = float(np.clip(u, 0, self.camera.W - 1))
        v = float(np.clip(v, 0, self.camera.H - 1))
        return int(u), int(v)


# =======================
# SAFE ROI EXTRACTOR
# =======================
def safe_roi(frame, x1, y1, x2, y2):
    """
    Returns cropped ROI only if it has valid non-zero dimensions.
    Returns None otherwise to avoid the empty-frame cvtColor crash.
    """
    H, W = frame.shape[:2]
    x1 = int(max(0, min(x1, W - 1)))
    y1 = int(max(0, min(y1, H - 1)))
    x2 = int(max(0, min(x2, W)))
    y2 = int(max(0, min(y2, H)))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi


# =======================
# DETECTION FUNCTIONS
# =======================
def detect_motion(frame, prev_frame):
    """Returns [] immediately if either frame is None or empty."""
    if prev_frame is None or frame is None:
        return []
    if frame.size == 0 or prev_frame.size == 0:
        return []
    if frame.shape != prev_frame.shape:
        return []

    g1 = cv2.GaussianBlur(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (21,21), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY), (21,21), 0)

    diff = cv2.absdiff(g1, g2)
    _, th = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    th = cv2.dilate(th, kernel, iterations=5)
    th = cv2.erode(th,  kernel, iterations=3)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 400 < area < 50000:
            x, y, w, h = cv2.boundingRect(c)
            ar = h / float(w) if w > 0 else 0
            if 1.1 < ar < 5.5 and h > 18 and w > 12:
                pw, ph = int(w * 0.6), int(h * 0.4)
                boxes.append((max(0, x-pw), max(0, y-ph), w+2*pw, h+2*ph))
    return boxes


def detect_bg(frame, bg):
    if frame is None or frame.size == 0:
        return []

    fg = bg.apply(frame, learningRate=0.012)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 500 < area < 45000:
            x, y, w, h = cv2.boundingRect(c)
            ar = h / float(w) if w > 0 else 0
            if 1.1 < ar < 5.5 and h > 22 and w > 12:
                pw, ph = int(w * 0.6), int(h * 0.4)
                boxes.append((max(0, x-pw), max(0, y-ph), w+2*pw, h+2*ph))
    return boxes


def detect_hog(frame, hog):
    if frame is None or frame.size == 0:
        return []
    try:
        rects, _ = hog.detectMultiScale(frame, winStride=(8,8), padding=(4,4), scale=1.05)
        return list(rects)
    except:
        return []


def combine_detections(boxes, max_dist=180):
    if not boxes:
        return []
    merged, used = [], set()
    for i, (x1,y1,w1,h1) in enumerate(boxes):
        if i in used:
            continue
        cx1, cy1 = x1+w1//2, y1+h1//2
        group = [(x1,y1,w1,h1)]
        for j, (x2,y2,w2,h2) in enumerate(boxes[i+1:], i+1):
            if j in used:
                continue
            if np.hypot(cx1-(x2+w2//2), cy1-(y2+h2//2)) < max_dist:
                group.append((x2,y2,w2,h2))
                used.add(j)
        ax  = [b[0]      for b in group]
        ay  = [b[1]      for b in group]
        ax2 = [b[0]+b[2] for b in group]
        ay2 = [b[1]+b[3] for b in group]
        merged.append((min(ax), min(ay), max(ax2)-min(ax), max(ay2)-min(ay)))
        used.add(i)
    return merged


def compute_iou_boxes(b1, b2):
    x1,y1,w1,h1 = b1[0]-b1[2]//2, b1[1]-b1[3]//2, b1[2], b1[3]
    x2,y2,w2,h2 = b2[0]-b2[2]//2, b2[1]-b2[3]//2, b2[2], b2[3]
    xi1 = max(x1, x2); yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2); yi2 = min(y1+h1, y2+h2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter/union if union > 0 else 0


def detect_occlusion(trackers, pred_boxes, iou_thresh=0.25):
    occluded = [False] * len(trackers)
    for i in range(len(trackers)):
        for j in range(i+1, len(trackers)):
            if compute_iou_boxes(pred_boxes[i], pred_boxes[j]) > iou_thresh:
                occluded[i] = True
                occluded[j] = True
    return occluded


def assign_detections_to_trackers(trackers, detections, max_dist=300):
    assignments, assigned = [], set()
    for t_idx, t in enumerate(trackers):
        tcx, tcy = t.center()
        best_dist, best_d = max_dist, -1
        for d_idx, (x,y,w,h) in enumerate(detections):
            if d_idx in assigned:
                continue
            d = np.hypot(x+w//2 - tcx, y+h//2 - tcy)
            if d < best_dist:
                best_dist, best_d = d, d_idx
        if best_d >= 0:
            assignments.append((t_idx, detections[best_d]))
            assigned.add(best_d)
    return assignments


# ── Colors ──
PRED_COLORS   = [(0,0,255), (255,128,0)]
DET_COLORS    = [(0,255,0), (0,255,255)]
OCCLUDE_COLOR = (0,165,255)
MAX_OBJECTS   = 2
MAX_LOST      = 120
MIN_CONF      = 2


# =======================
# MAIN
# =======================
print("=" * 60)
print("3D UKF DUAL TRACKER - OCCLUSION VIDEO  (FIXED)")
print("=" * 60)

VIDEO_FILE = "Twopersonocllusion.mp4"
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"❌ Cannot open '{VIDEO_FILE}'")
    exit()

ret, frame = cap.read()
if not ret:
    print("❌ Cannot read first frame")
    exit()

prev_frame = frame.copy()
H, W = frame.shape[:2]
fps  = cap.get(cv2.CAP_PROP_FPS)
print(f"📹 {W}x{H} @ {fps:.1f} FPS  |  🎮 ESC=Quit  SPACE=Pause  R=Reset\n")

camera = CameraModel(W, H)
priors = PedestrianPriors()
bg     = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=18, detectShadows=False)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

trackers    = []
next_id     = 0
consecutive = 0

total_frames          = 0
frames_with_detection = [0] * MAX_OBJECTS
accurate_detections   = [0] * MAX_OBJECTS
occlusion_events      = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    total_frames += 1

    # ── Full-frame detection ──
    all_dets = combine_detections(
        detect_motion(frame, prev_frame) + detect_bg(frame, bg), max_dist=200)
    all_dets = [b for b in all_dets if b[2] > 20 and b[3] > 45]

    # ── Init new trackers ──
    if len(trackers) < MAX_OBJECTS and all_dets:
        consecutive += 1
        if consecutive >= MIN_CONF:
            existing = [t.center() for t in trackers]
            for (x, y, w, h) in sorted(all_dets, key=lambda b: b[2]*b[3], reverse=True):
                if len(trackers) >= MAX_OBJECTS:
                    break
                cx, cy = x+w//2, y+h//2
                if not any(np.hypot(cx-ex, cy-ey) < 80 for ex, ey in existing):
                    trackers.append(UKFTracker3D(cx, cy, h, next_id, camera,
                                                 dt=1.0/fps if fps > 0 else 0.033))
                    existing.append((cx, cy))
                    print(f"✅ Frame {total_frames}: 3D UKF Obj {next_id} initialised")
                    next_id += 1
    else:
        if not all_dets:
            consecutive = 0

    if not trackers:
        prev_frame = frame.copy()
        cv2.imshow("3D UKF Occlusion Tracker", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break
        continue

    # ── Predict all ──
    predictions = [t.predict() for t in trackers]   # (pu, pv, depth)

    # ── Build pred boxes for occlusion check ──
    pred_boxes = []
    for t_idx, (pu, pv, _) in enumerate(predictions):
        bw, bh = trackers[t_idx].get_predicted_bbox_size()
        bw = max(40, min(bw, 200))
        bh = max(80, min(bh, 350))
        pred_boxes.append((pu, pv, bw, bh))

    # ── Occlusion detection ──
    occluded_flags = detect_occlusion(trackers, pred_boxes, iou_thresh=0.25)
    for t_idx, t in enumerate(trackers):
        was_occ = t.is_occluded
        t.is_occluded = occluded_flags[t_idx]
        if t.is_occluded:
            t.occluded_frames += 1
            if not was_occ:
                occlusion_events += 1

    # ── ROI detections (wider when occluded) ──
    all_roi_dets = []
    for t_idx, (pu, pv, _) in enumerate(predictions):
        t = trackers[t_idx]
        base_roi = 380 if t.is_occluded else 280
        ROI = min(base_roi + t.lost_frames * 25, 700)

        x1 = int(max(0, pu - ROI))
        y1 = int(max(0, pv - ROI))
        x2 = int(min(W,  pu + ROI))
        y2 = int(min(H,  pv + ROI))

        # ── SAFE ROI extraction ──
        roi      = safe_roi(frame,      x1, y1, x2, y2)
        prev_roi = safe_roi(prev_frame, x1, y1, x2, y2)

        local = []
        if roi is not None:
            local += detect_bg(roi, bg)
            if prev_roi is not None and roi.shape == prev_roi.shape:
                local += detect_motion(roi, prev_roi)
            # HOG during/after occlusion
            if t.is_occluded or t.lost_frames > 10:
                local += detect_hog(roi, hog)

        # Convert local coords → global
        all_roi_dets.extend([(x1+x, y1+y, w, h) for (x, y, w, h) in local])

    all_roi_dets = combine_detections(all_roi_dets)
    assignments  = assign_detections_to_trackers(trackers, all_roi_dets)
    assigned_ids = set()

    for (t_idx, (x, y, w, h)) in assignments:
        t = trackers[t_idx]
        pu, pv, _ = predictions[t_idx]
        cx, cy = x+w//2, y+h//2
        dist   = np.hypot(cx-pu, cy-pv)
        factor = 0.85 if t.is_occluded else 0.65
        dist_threshold = factor * min(280 + t.lost_frames*25, 700)

        if dist < dist_threshold:
            t.update(cx, cy, h)
            t.lost_frames = 0
            assigned_ids.add(t_idx)

            obj_idx = t_idx % MAX_OBJECTS
            frames_with_detection[obj_idx] += 1
            if dist < 45:
                accurate_detections[obj_idx] += 1

            cv2.rectangle(frame, (x,y), (x+w,y+h), DET_COLORS[obj_idx], 3)
            cv2.putText(frame, f"UKF Obj {t.id} DETECT",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DET_COLORS[obj_idx], 2)

    # ── Draw prediction boxes + trails ──
    for t_idx, t in enumerate(trackers):
        pu, pv, depth = predictions[t_idx]
        obj_idx = t_idx % MAX_OBJECTS
        bw = pred_boxes[t_idx][2]
        bh = pred_boxes[t_idx][3]

        if t_idx not in assigned_ids:
            t.lost_frames += 1

        color = OCCLUDE_COLOR if t.is_occluded else PRED_COLORS[obj_idx]

        if t.is_occluded:
            label = f"UKF Obj {t.id} OCCLUDED ({t.occluded_frames}f) Z={depth:.1f}m"
        elif t_idx not in assigned_ids:
            label = f"UKF Obj {t.id} PRED ({t.lost_frames}f) Z={depth:.1f}m"
        else:
            label = f"UKF Obj {t.id} Z={depth:.1f}m"

        # Clamp box drawing coords
        bx1 = max(0, pu-bw//2); by1 = max(0, pv-bh//2)
        bx2 = min(W, pu+bw//2); by2 = min(H, pv+bh//2)
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), color, 3)
        cv2.putText(frame, label, (bx1, max(10, by1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # 3D velocity
        st  = t.get_3d_state()
        vel = np.sqrt(st[3]**2 + st[4]**2 + st[5]**2)
        cv2.putText(frame, f"v={vel:.2f}m/s",
                    (bx1, min(H-5, by2+16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Trail
        pts = list(t.trail)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], color, 2)

    # ── Remove dead trackers ──
    before   = len(trackers)
    trackers = [t for t in trackers if t.lost_frames <= MAX_LOST]
    if len(trackers) < before:
        print(f"⚠️  Frame {total_frames}: Lost a tracker")
        consecutive = 0

    # ── HUD ──
    occ_now = sum(1 for t in trackers if t.is_occluded)
    cv2.rectangle(frame, (0,0), (W, 100), (40,40,40), -1)
    cv2.putText(frame,
                f"Frame:{total_frames} | 3D UKF {len(trackers)}/{MAX_OBJECTS} | Occluded:{occ_now} | Events:{occlusion_events}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)

    line2 = ""
    if trackers:
        st = trackers[0].get_3d_state()
        line2 += f"Obj0: X={st[0]:.1f} Z={st[2]:.1f}m"
    if len(trackers) > 1:
        st = trackers[1].get_3d_state()
        line2 += f"  |  Obj1: X={st[0]:.1f} Z={st[2]:.1f}m"
    cv2.putText(frame, line2, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,255,200), 1)
    cv2.putText(frame, "Orange=Occluded  Pred=Dashed  Green/Yellow=Detected",
                (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)

    prev_frame = frame.copy()
    cv2.imshow("3D UKF Occlusion Tracker", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
    elif key == ord(' '):
        cv2.waitKey(0)
    elif key == ord('r'):
        trackers = []; consecutive = 0; next_id = 0; occlusion_events = 0
        bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=18, detectShadows=False)
        print("🔄 RESET")

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*60}\nFINAL STATISTICS\n{'='*60}")
print(f"Total Frames:     {total_frames}")
print(f"Occlusion Events: {occlusion_events}")
for i in range(MAX_OBJECTS):
    print(f"\nObject {i+1}:")
    print(f"  Detection Rate:     {(frames_with_detection[i]/total_frames)*100:.1f}%")
    if frames_with_detection[i] > 0:
        print(f"  Detection Accuracy: {(accurate_detections[i]/frames_with_detection[i])*100:.1f}%")
print("=" * 60)