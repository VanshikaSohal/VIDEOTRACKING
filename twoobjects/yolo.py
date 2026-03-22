"""
YOLO + KALMAN TRACKER - TWO OBJECTS
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# =======================
# KALMAN TRACKER
# =======================
class SimpleKalmanTracker:
    def __init__(self, x, y, w, h, tracker_id):
        self.id = tracker_id
        self.lost_frames = 0
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
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)

        cx = x + w // 2
        cy = y + h // 2
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.w = w
        self.h = h

    def predict(self):
        pred = self.kf.predict()
        cx = int(pred[0][0])     # ✅ pred[0][0] gets the scalar value
        cy = int(pred[1][0])
        x = cx - self.w // 2
        y = cy - self.h // 2
        self.trail.append((cx, cy))
        return x, y, self.w, self.h

    def update(self, x, y, w, h):
        self.w = w
        self.h = h
        cx = x + w // 2
        cy = y + h // 2
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(measurement)

    def center(self):
       s = self.kf.statePost
       return int(s[0][0]), int(s[1][0]) 


# =======================
# ASSIGNMENT
# =======================
def assign_detections_to_trackers(trackers, detections, max_dist=250):
    """
    Greedy nearest-distance assignment.
    Returns list of (tracker_idx, detection) pairs.
    """
    assignments = []
    assigned_dets = set()

    for t_idx, t in enumerate(trackers):
        tcx, tcy = t.center()
        best_dist, best_d_idx = max_dist, -1

        for d_idx, (x, y, w, h) in enumerate(detections):
            if d_idx in assigned_dets:
                continue
            dcx, dcy = x + w//2, y + h//2
            d = np.hypot(dcx - tcx, dcy - tcy)
            if d < best_dist:
                best_dist, best_d_idx = d, d_idx

        if best_d_idx >= 0:
            assignments.append((t_idx, detections[best_d_idx]))
            assigned_dets.add(best_d_idx)

    return assignments


# Colors per object
PRED_COLORS = [(0, 255, 255), (255, 128, 0)]   # Yellow, Orange  → prediction
DET_COLORS  = [(0, 255, 0),   (0, 128, 255)]   # Green,  Blue    → detection
MAX_OBJECTS = 2
MAX_LOST    = 30


# =======================
# MAIN
# =======================
def main():
    # ============== CONFIGURATION ==============
    VIDEO_PATH  = "twoobjectscurved.mp4"
    YOLO_MODEL  = "yolov8n.pt"
    CONFIDENCE  = 0.5
    TRACK_CLASS = 0    # 0=person, 2=car, etc.
    # ===========================================

    print("🚀 Starting YOLO + Kalman Dual Tracker...")
    print(f"📹 Video: {VIDEO_PATH}")
    print(f"🎯 Tracking class: {TRACK_CLASS}  |  Max objects: {MAX_OBJECTS}")

    model = YOLO(YOLO_MODEL)
    print("✅ Model loaded!")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Cannot open video '{VIDEO_PATH}'")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"📊 {width}x{height} @ {fps} FPS\n▶️  Press 'q' or ESC to quit\n")

    trackers   = []   # list of active SimpleKalmanTracker
    next_id    = 0

    total_frames          = 0
    frames_with_detection = [0] * MAX_OBJECTS
    accurate_detections   = [0] * MAX_OBJECTS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n🏁 End of video")
            break

        total_frames += 1

        # ─────────────────────────────────────────
        # STEP 1: Full-frame YOLO detection
        # ─────────────────────────────────────────
        results = model(frame, conf=CONFIDENCE, verbose=False)
        all_dets = []   # list of (x, y, w, h)

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == TRACK_CLASS:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    all_dets.append((int(x1), int(y1),
                                     int(x2-x1), int(y2-y1)))

        # ─────────────────────────────────────────
        # STEP 2: Initialise new trackers (up to MAX_OBJECTS)
        # ─────────────────────────────────────────
        if len(trackers) < MAX_OBJECTS and all_dets:
            existing = [t.center() for t in trackers]

            for (x, y, w, h) in all_dets:
                if len(trackers) >= MAX_OBJECTS:
                    break
                cx, cy = x + w//2, y + h//2
                # Don't create a tracker too close to an existing one
                if not any(np.hypot(cx-ex, cy-ey) < 80 for ex, ey in existing):
                    trackers.append(SimpleKalmanTracker(x, y, w, h, next_id))
                    existing.append((cx, cy))
                    print(f"✅ Object {next_id} initialised at frame {total_frames}")
                    next_id += 1

        # ─────────────────────────────────────────
        # STEP 3: Predict all trackers
        # ─────────────────────────────────────────
        predictions = [t.predict() for t in trackers]  # list of (x, y, w, h)

        # ─────────────────────────────────────────
        # STEP 4: ROI-based YOLO detection per tracker
        #         (more precise than full-frame dets)
        # ─────────────────────────────────────────
        roi_dets = []
        SEARCH = 150   # pixels around predicted box

        for t_idx, (px, py, pw, ph) in enumerate(predictions):
            rx1 = max(0, px - SEARCH)
            ry1 = max(0, py - SEARCH)
            rx2 = min(width,  px + pw + SEARCH)
            ry2 = min(height, py + ph + SEARCH)

            roi = frame[ry1:ry2, rx1:rx2]
            roi_results = model(roi, conf=CONFIDENCE, verbose=False)

            for result in roi_results:
                for box in result.boxes:
                    if int(box.cls[0]) == TRACK_CLASS:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # Convert back to full-frame coords
                        roi_dets.append((int(x1)+rx1, int(y1)+ry1,
                                         int(x2-x1),  int(y2-y1)))

        # De-duplicate ROI dets that are very close
        seen = []
        for d in roi_dets:
            cx, cy = d[0]+d[2]//2, d[1]+d[3]//2
            if not any(np.hypot(cx-sx, cy-sy) < 50 for sx, sy in seen):
                seen.append((cx, cy))
            else:
                roi_dets.remove(d)

        # ─────────────────────────────────────────
        # STEP 5: Assign detections → trackers
        # ─────────────────────────────────────────
        assignments  = assign_detections_to_trackers(trackers, roi_dets)
        assigned_ids = set()

        for (t_idx, (x, y, w, h)) in assignments:
            t = trackers[t_idx]
            tcx, tcy = t.center()
            dcx, dcy = x + w//2, y + h//2
            dist = np.hypot(dcx - tcx, dcy - tcy)

            t.update(x, y, w, h)
            t.lost_frames = 0
            assigned_ids.add(t_idx)

            obj_idx = t_idx % MAX_OBJECTS
            frames_with_detection[obj_idx] += 1
            if dist < 40:
                accurate_detections[obj_idx] += 1

            # GREEN / BLUE detection box
            cv2.rectangle(frame, (x, y), (x+w, y+h), DET_COLORS[obj_idx], 3)
            cv2.putText(frame, f"Obj {t.id} DETECTED",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, DET_COLORS[obj_idx], 2)

        # ─────────────────────────────────────────
        # STEP 6: Handle unmatched trackers (predict-only)
        # ─────────────────────────────────────────
        for t_idx, t in enumerate(trackers):
            px, py, pw, ph = predictions[t_idx]
            obj_idx = t_idx % MAX_OBJECTS
            color   = PRED_COLORS[obj_idx]

            if t_idx not in assigned_ids:
                t.lost_frames += 1
                # YELLOW / ORANGE prediction box
                cv2.rectangle(frame, (px, py), (px+pw, py+ph), color, 3)
                cv2.putText(frame, f"Obj {t.id} PREDICTING ({t.lost_frames})",
                            (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

            # Motion trail
            pts = list(t.trail)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 2)

        # ─────────────────────────────────────────
        # STEP 7: Remove dead trackers
        # ─────────────────────────────────────────
        before = len(trackers)
        trackers = [t for t in trackers if t.lost_frames <= MAX_LOST]
        if len(trackers) < before:
            print(f"❌ Lost a tracker at frame {total_frames}")

        # ─────────────────────────────────────────
        # STEP 8: HUD
        # ─────────────────────────────────────────
        status = f"Tracking {len(trackers)}/{MAX_OBJECTS} object(s)"
        cv2.putText(frame, f"Frame {total_frames} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.imshow("YOLO + Kalman Dual Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n⏹️  Stopped by user")
            break

    cap.release()
    cv2.destroyAllWindows()

    # ─── METRICS ───
    print(f"\n✅ Total Frames: {total_frames}")
    for i in range(MAX_OBJECTS):
        print(f"\nObject {i+1}:")
        print(f"  Detection Rate:     {(frames_with_detection[i]/total_frames)*100:.2f}%")
        if frames_with_detection[i] > 0:
            print(f"  Detection Accuracy: {(accurate_detections[i]/frames_with_detection[i])*100:.2f}%")
    print("\n👋 Done!")


if __name__ == "__main__":
    main()