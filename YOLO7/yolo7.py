import cv2
import torch
import numpy as np
import sys
import os
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS  – change only these two lines if needed
# ─────────────────────────────────────────────
YOLOV7_REPO  = r'C:\Users\vansh\.cache\torch\hub\WongKinYiu_yolov7_main'
WEIGHTS_PATH = r'C:\Users\vansh\.cache\torch\hub\WongKinYiu_yolov7_main\yolov7.pt'
VIDEO_PATH   = r'C:\Users\vansh\Downloads\FOLDER\MAINPROJECTS\KALMANFILTER\VIDEOS\realvideo.mp4'
CONFIDENCE   = 0.5
IOU_THRESH   = 0.45
IMG_SIZE     = 640
# ─────────────────────────────────────────────

# 1. Add YOLOv7 repo to path
sys.path.insert(0, YOLOV7_REPO)

print(f"NumPy  : {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"OpenCV : {cv2.__version__}")

# 2. Check weights file exists
if not os.path.exists(WEIGHTS_PATH):
    print(f"\n[ERROR] yolov7.pt not found at:\n  {WEIGHTS_PATH}")
    print("\nDownload it from:")
    print("  https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
    print(f"\nThen place it in: {YOLOV7_REPO}")
    sys.exit(1)

# 3. Import YOLOv7 utilities AFTER sys.path is set
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

# 4. Load model  (weights_only=False fixes PyTorch 2.6+ error)
print("\nLoading YOLOv7 model...")
device = select_device('0' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

try:
    # Primary method – works with PyTorch >= 2.6
    ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    model = (ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt).float().eval()
    print("✓ Model loaded (direct torch.load)")

except Exception as e1:
    print(f"  Direct load failed ({e1}), trying attempt_load ...")
    try:
        # Fallback – monkey-patch torch.load so attempt_load also passes weights_only=False
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
        model = attempt_load(WEIGHTS_PATH, map_location=device)
        torch.load = _orig_load
        print("✓ Model loaded (attempt_load fallback)")
    except Exception as e2:
        print(f"[ERROR] Both load methods failed.\n  {e2}")
        sys.exit(1)

# 5. Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

# ─────────────────────────────────────────────
# Kalman Tracker
# ─────────────────────────────────────────────
class SimpleKalmanTracker:
    """
    4-state (cx, cy, vx, vy) → 2-measurement (cx, cy) Kalman filter.
    Tracks centre of bounding box; remembers width & height separately.
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
        self.kf.processNoiseCov  = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.w, self.h = w, h

    def update(self, x1, y1, w, h):
        cx, cy = x1 + w // 2, y1 + h // 2
        self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        self.w, self.h = w, h   # keep latest box size

    def predict(self):
        """Returns (x1, y1, w, h) of predicted box."""
        p = self.kf.predict()
        cx, cy = int(p[0, 0]), int(p[1, 0])
        return cx - self.w // 2, cy - self.h // 2, self.w, self.h


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
tracker    = None
frame_idx  = 0
stride     = int(model.stride.max()) if hasattr(model, 'stride') else 32

print("\nRunning… press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_idx += 1
    img0 = frame.copy()

    # ── Pre-process ──────────────────────────
    img = letterbox(img0, IMG_SIZE, stride=stride)[0]           # resize + pad
    img = img[:, :, ::-1].transpose(2, 0, 1)                    # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # ── Inference ────────────────────────────
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, CONFIDENCE, IOU_THRESH)[0]

    # ── Parse detections ─────────────────────
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()

        for *xyxy, conf, cls in pred:
            if int(cls) != 0:          # 0 = person
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1

            # GREEN detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Det {conf:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if tracker is None:
                tracker = SimpleKalmanTracker(x1, y1, w, h)
            else:
                tracker.update(x1, y1, w, h)
            break   # track only the first / highest-conf person

    # ── Kalman prediction (BLUE box) ─────────
    if tracker:
        px, py, pw, ph = tracker.predict()
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 100, 0), 2)
        cv2.putText(frame, "Kalman Predict",
                    (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # ── HUD ──────────────────────────────────
    cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLOv7 + Kalman Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done – processed {frame_idx} frames.")