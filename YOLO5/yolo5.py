import cv2
import numpy as np
import torch

# ─────────────────────────────────────────────
# PATHS – change if needed
# ─────────────────────────────────────────────
VIDEO_PATH    = r'C:\Users\vansh\Downloads\FOLDER\MAINPROJECTS\KALMANFILTER\VIDEOS\realvideo.mp4'
WEIGHTS_PATH  = r'C:\Users\vansh\Downloads\FOLDER\MAINPROJECTS\KALMANFILTER\YOLO5\yolov5n.pt'
CONFIDENCE    = 0.5
DETECT_EVERY  = 3      # run YOLO every N frames (speed boost)
IMG_SIZE      = 416    # smaller = faster (was 640)
# ─────────────────────────────────────────────

print(f"PyTorch : {torch.__version__}")
print(f"OpenCV  : {cv2.__version__}")

# ── Load model ───────────────────────────────
# FIX 1: Load from local .pt file instead of always downloading from hub
# FIX 2: force_reload=False avoids re-download every run
print("Loading YOLOv5 model...")
try:
    # Try local weights first (fastest, no internet needed)
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=WEIGHTS_PATH,
        force_reload=False,
        verbose=False
    )
    print(f"✓ Loaded local weights: {WEIGHTS_PATH}")
except Exception as e:
    print(f"  Local load failed ({e})")
    print("  Downloading yolov5n from hub...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, verbose=False)
    print("✓ Downloaded yolov5n from hub")

# FIX 3: Set conf, iou, image size, and device properly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.conf  = CONFIDENCE
model.iou   = 0.45
model.classes = [0]       # only detect 'person' (class 0) — faster!
model.eval()
print(f"Device  : {device}")


# ── Kalman Tracker ───────────────────────────
class SimpleKalmanTracker:
    """
    Tracks bounding-box centre (cx, cy) with velocity.
    FIX 4: Added kf.correct() in update so tracker actually learns from detections.
    FIX 5: processNoiseCov & measurementNoiseCov set for stable tracking.
    """
    def __init__(self, x1, y1, w, h):
        cx, cy = x1 + w // 2, y1 + h // 2
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.w, self.h = w, h

    def update(self, x1, y1, w, h):
        """Call this when a new detection arrives."""
        cx, cy = x1 + w // 2, y1 + h // 2
        self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        self.w, self.h = w, h   # update box size

    def predict(self):
        """Returns (x1, y1, w, h) — call every frame."""
        p  = self.kf.predict()
        cx = int(p[0, 0])
        cy = int(p[1, 0])
        return cx - self.w // 2, cy - self.h // 2, self.w, self.h


# ── Open video ───────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    exit(1)

tracker    = None
frame_idx  = 0
print("\nRunning… press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_idx += 1

    # FIX 6: Run YOLO only every N frames (big speed boost)
    # Kalman predict() fills the gap every frame — still smooth
    if frame_idx % DETECT_EVERY == 0:
        # FIX 7: Pass size= so model doesn't guess input resolution
        results = model(frame, size=IMG_SIZE)

        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:
                bx, by = int(x1), int(y1)
                bw, bh = int(x2 - x1), int(y2 - y1)

                # GREEN = raw detection box
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"Det {conf:.2f}",
                            (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                if tracker is None:
                    tracker = SimpleKalmanTracker(bx, by, bw, bh)
                else:
                    # FIX 4: actually update the filter with new measurement
                    tracker.update(bx, by, bw, bh)
                break   # track first/best person only

    # BLUE = Kalman prediction (runs every frame)
    if tracker:
        px, py, pw, ph = tracker.predict()
        # clamp to frame bounds
        px = max(0, px);  py = max(0, py)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 100, 0), 2)
        cv2.putText(frame, "Kalman",
                    (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # HUD
    cv2.putText(frame, f"Frame {frame_idx} | {device.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLOv5 + Kalman Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done – processed {frame_idx} frames.")