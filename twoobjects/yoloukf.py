"""
YOLO + UNSCENTED KALMAN FILTER (UKF) - TWO OBJECTS
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# =======================
# UKF TRACKER
# =======================
def fx(x, dt):
    """State transition"""
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ])
    return F @ x

def hx(x):
    """Measurement function"""
    return np.array([x[0], x[1]])

class UKFTracker:
    def __init__(self, x, y, w, h, tracker_id, dt=1.0):
        self.id          = tracker_id
        self.lost_frames = 0
        self.trail       = deque(maxlen=40)
        self.w = w
        self.h = h

        points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2., kappa=0)
        self.ukf = UKF(dim_x=4, dim_z=2,
                       fx=lambda x, dt: fx(x, dt),
                       hx=hx, dt=dt, points=points)

        cx = x + w // 2
        cy = y + h // 2
        self.ukf.x  = np.array([cx, cy, 0, 0], dtype=float)
        self.ukf.P *= 10.0
        self.ukf.R  = np.eye(2) * 5.0
        self.ukf.Q  = np.eye(4) * 0.03

    def predict(self):
        self.ukf.predict()
        cx = int(self.ukf.x[0])
        cy = int(self.ukf.x[1])
        self.trail.append((cx, cy))
        x = cx - self.w // 2
        y = cy - self.h // 2
        return x, y, self.w, self.h

    def update(self, x, y, w, h):
        self.w = w
        self.h = h
        cx = x + w // 2
        cy = y + h // 2
        self.ukf.update(np.array([float(cx), float(cy)]))

    def center(self):
        return int(self.ukf.x[0]), int(self.ukf.x[1])


# =======================
# ASSIGNMENT
# =======================
def assign_detections_to_trackers(trackers, detections, max_dist=250):
    assignments   = []
    assigned_dets = set()
    for t_idx, t in enumerate(trackers):
        tcx, tcy = t.center()
        best_dist, best_d_idx = max_dist, -1
        for d_idx, (x, y, w, h) in enumerate(detections):
            if d_idx in assigned_dets:
                continue
            d = np.hypot(x+w//2 - tcx, y+h//2 - tcy)
            if d < best_dist:
                best_dist, best_d_idx = d, d_idx
        if best_d_idx >= 0:
            assignments.append((t_idx, detections[best_d_idx]))
            assigned_dets.add(best_d_idx)
    return assignments


# Colors: [Obj1, Obj2]
PRED_COLORS = [(0, 255, 255), (255, 128, 0)]
DET_COLORS  = [(0, 255, 0),   (0, 128, 255)]
MAX_OBJECTS = 2
MAX_LOST    = 30


# =======================
# MAIN
# =======================
def main():
    VIDEO_PATH  = "twoobjectscurved.mp4"
    YOLO_MODEL  = "yolov8n.pt"
    CONFIDENCE  = 0.5
    TRACK_CLASS = 0

    print("🚀 Starting YOLO + UKF Dual Tracker...")
    model = YOLO(YOLO_MODEL)
    print("✅ Model loaded!")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Cannot open video '{VIDEO_PATH}'")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📊 {width}x{height} @ {int(cap.get(cv2.CAP_PROP_FPS))} FPS")
    print("▶️  Press 'q' or ESC to quit\n")

    trackers = []
    next_id  = 0

    total_frames          = 0
    frames_with_detection = [0] * MAX_OBJECTS
    accurate_detections   = [0] * MAX_OBJECTS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n🏁 End of video")
            break
        total_frames += 1

        # ── Full-frame YOLO detection ──
        results  = model(frame, conf=CONFIDENCE, verbose=False)
        all_dets = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == TRACK_CLASS:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    all_dets.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))

        # ── Init new trackers ──
        if len(trackers) < MAX_OBJECTS and all_dets:
            existing = [t.center() for t in trackers]
            for (x, y, w, h) in all_dets:
                if len(trackers) >= MAX_OBJECTS:
                    break
                cx, cy = x+w//2, y+h//2
                if not any(np.hypot(cx-ex, cy-ey) < 80 for ex, ey in existing):
                    trackers.append(UKFTracker(x, y, w, h, next_id))
                    existing.append((cx, cy))
                    print(f"✅ UKF Object {next_id} initialised at frame {total_frames}")
                    next_id += 1

        # ── Predict all ──
        predictions = [t.predict() for t in trackers]

        # ── ROI-based YOLO detection ──
        roi_dets = []
        SEARCH   = 150
        for t_idx, (px, py, pw, ph) in enumerate(predictions):
            rx1 = max(0, px - SEARCH);       ry1 = max(0, py - SEARCH)
            rx2 = min(width,  px+pw+SEARCH);  ry2 = min(height, py+ph+SEARCH)
            roi = frame[ry1:ry2, rx1:rx2]
            for result in model(roi, conf=CONFIDENCE, verbose=False):
                for box in result.boxes:
                    if int(box.cls[0]) == TRACK_CLASS:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        roi_dets.append((int(x1)+rx1, int(y1)+ry1,
                                         int(x2-x1),  int(y2-y1)))

        # De-duplicate
        seen, clean = [], []
        for d in roi_dets:
            cx, cy = d[0]+d[2]//2, d[1]+d[3]//2
            if not any(np.hypot(cx-sx, cy-sy) < 50 for sx, sy in seen):
                clean.append(d); seen.append((cx, cy))
        roi_dets = clean

        # ── Assign ──
        assignments  = assign_detections_to_trackers(trackers, roi_dets)
        assigned_ids = set()

        for (t_idx, (x, y, w, h)) in assignments:
            t = trackers[t_idx]
            tcx, tcy = t.center()
            dist = np.hypot(x+w//2 - tcx, y+h//2 - tcy)
            t.update(x, y, w, h)
            t.lost_frames = 0
            assigned_ids.add(t_idx)

            obj_idx = t_idx % MAX_OBJECTS
            frames_with_detection[obj_idx] += 1
            if dist < 40:
                accurate_detections[obj_idx] += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), DET_COLORS[obj_idx], 3)
            cv2.putText(frame, f"UKF Obj {t.id} DETECTED",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, DET_COLORS[obj_idx], 2)

        # ── Unmatched trackers ──
        for t_idx, t in enumerate(trackers):
            px, py, pw, ph = predictions[t_idx]
            obj_idx = t_idx % MAX_OBJECTS
            color   = PRED_COLORS[obj_idx]
            if t_idx not in assigned_ids:
                t.lost_frames += 1
                cv2.rectangle(frame, (px, py), (px+pw, py+ph), color, 3)
                cv2.putText(frame, f"UKF Obj {t.id} PREDICTING ({t.lost_frames})",
                            (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Trail
            pts = list(t.trail)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 2)

        # ── Remove dead trackers ──
        before   = len(trackers)
        trackers = [t for t in trackers if t.lost_frames <= MAX_LOST]
        if len(trackers) < before:
            print(f"❌ Lost a tracker at frame {total_frames}")

        # ── HUD ──
        cv2.putText(frame,
                    f"Frame {total_frames} | UKF Tracking {len(trackers)}/{MAX_OBJECTS}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("YOLO + UKF Dual Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n⏹️  Stopped by user"); break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Total Frames: {total_frames}")
    for i in range(MAX_OBJECTS):
        print(f"\nObject {i+1}:")
        print(f"  Detection Rate:     {(frames_with_detection[i]/total_frames)*100:.2f}%")
        if frames_with_detection[i] > 0:
            print(f"  Detection Accuracy: {(accurate_detections[i]/frames_with_detection[i])*100:.2f}%")
    print("\n👋 Done!")

if __name__ == "__main__":
    main()