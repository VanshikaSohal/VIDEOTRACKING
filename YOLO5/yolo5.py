import cv2
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.5

cap = cv2.VideoCapture("realvideo.mp4")
tracker = None

class SimpleKalmanTracker:
    def __init__(self, x, y, w, h):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.statePost = np.array([[x+w//2],[y+h//2],[0],[0]], np.float32)
        self.w, self.h = w, h

    def predict(self):
        p = self.kf.predict()
        return int(p[0]-self.w//2), int(p[1]-self.h//2), self.w, self.h

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for det in results.xyxy[0]:
        x1,y1,x2,y2,conf,cls = det
        if int(cls)==0:
            tracker = SimpleKalmanTracker(int(x1),int(y1),int(x2-x1),int(y2-y1))
            break

    if tracker:
        x,y,w,h = tracker.predict()
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("YOLOv5 + Kalman", frame)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
