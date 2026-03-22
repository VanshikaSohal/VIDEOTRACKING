import cv2
import torch
import numpy as np
import sys

# Local yolov7 repo use karo
sys.path.insert(0, r'C:\Users\vansh\.cache\torch\hub\WongKinYiu_yolov7_main')
from models.experimental import attempt_load

# FIXED: torch.hub hataya, local .pt use kar raha hai
model = attempt_load(r'C:\PROJECT\KALMANFILTER\YOLO7\yolov7.pt', map_location='cpu')
model.eval()

cap = cv2.VideoCapture(r'C:\PROJECT\KALMANFILTER\VIDEOS\realvideo.mp4')

class SimpleKalmanTracker:
    def __init__(self, x, y, w, h):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
        self.w, self.h = w, h

    def update(self, x, y):
        self.kf.correct(np.array([[np.float32(x)],[np.float32(y)]]))

    def predict(self):
        p = self.kf.predict()
        return int(p[0]), int(p[1]), self.w, self.h

tracker = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FIXED: yolov7 ko sahi input format chahiye
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        results = model(img)

    # FIXED: output sahi se parse kiya
    preds = results[0] if isinstance(results, tuple) else results
    preds = preds[preds[:, 4] > 0.5]  # confidence filter

    for d in preds:
        x1, y1, x2, y2, conf, cls = d[:6]
        if int(cls) == 0:
            cx, cy = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)

            if tracker is None:
                tracker = SimpleKalmanTracker(cx, cy, w, h)
            else:
                tracker.update(cx, cy)
                tracker.w, tracker.h = w, h
            break

    if tracker:
        x, y, w, h = tracker.predict()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("YOLOv7 + Kalman", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()