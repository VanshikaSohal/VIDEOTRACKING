import cv2
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF

# EKF Tracker
def fx(x, dt):
    """State transition function for constant velocity model"""
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return F @ x

def hx(x):
    """Measurement function: return position only"""
    return np.array([x[0], x[1]])

def H_jacobian(x):
    """Jacobian of measurement function"""
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return H

class EKFTracker:
    def __init__(self, x, y, dt=1.0):
        self.dt = dt
        self.ekf = EKF(dim_x=4, dim_z=2)
        
        # initial state [x, y, vx, vy]
        self.ekf.x = np.array([x, y, 0, 0], dtype=float)
        
        # state transition
        self.ekf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # measurement matrix
        self.ekf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # uncertainties
        self.ekf.P *= 10.0             # initial state covariance
        self.ekf.R = np.eye(2) * 6.0   # measurement noise
        self.ekf.Q = np.eye(4) * 0.03  # process noise

    def predict(self):
        self.ekf.predict()
        return int(self.ekf.x[0]), int(self.ekf.x[1])

    def update(self, x, y):
       z = np.array([x, y])
       self.ekf.update(z, HJacobian=H_jacobian, Hx=hx)

# DETECTION FUNCTIONS
def detect_motion(frame, prev_frame):
    if prev_frame is None or frame.shape != prev_frame.shape:
        return []

    g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(g1, g2)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, None, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        if cv2.contourArea(c) > 1200:
            boxes.append(cv2.boundingRect(c))
    return boxes

def detect_bg(frame, bg):
    fg = bg.apply(frame)
    fg = cv2.medianBlur(fg, 5)

    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        if cv2.contourArea(c) > 1500:
            boxes.append(cv2.boundingRect(c))
    return boxes

def detect_hog(frame, hog):
    rects, _ = hog.detectMultiScale(frame, winStride=(8, 8))
    return list(rects)

# =======================
# MAIN
# =======================
cap = cv2.VideoCapture("realcurvedvideo.mp4")
ret, frame = cap.read()
if not ret:
    print("Cannot read video file")
    exit()

prev_frame = frame.copy()
H, W = frame.shape[:2]

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25)

tracker = None
tracking = False
lost_frames = 0

total_frames = 0
frames_with_detection = 0
accurate_detections = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    used_measurement = False
    detection_bbox = None

    if not tracking:
        dets = detect_motion(frame, prev_frame) + detect_bg(frame, bg) + detect_hog(frame, hog)

        if dets:
            # Take largest detected object
            x, y, w, h = max(dets, key=lambda b: b[2]*b[3])
            cx, cy = x + w//2, y + h//2
            tracker = EKFTracker(cx, cy)
            tracking = True
            lost_frames = 0

    else:
        pred_x, pred_y = tracker.predict()

        # dynamic ROI grows if object lost for several frames
        ROI = min(250 + lost_frames * 20, 600)
        x1, y1 = max(0, pred_x-ROI), max(0, pred_y-ROI)
        x2, y2 = min(W, pred_x+ROI), min(H, pred_y+ROI)

        roi = frame[y1:y2, x1:x2]
        prev_roi = prev_frame[y1:y2, x1:x2]

        dets = detect_motion(roi, prev_roi) + detect_bg(roi, bg) + detect_hog(roi, hog)

        best_dist = 1e9
        best_center = None
        best_box = None

        for (x, y, w, h) in dets:
            cx = x1 + x + w//2
            cy = y1 + y + h//2
            d = np.hypot(cx - pred_x, cy - pred_y)

            if d < best_dist:
                best_dist = d
                best_center = (cx, cy)
                best_box = (x1+x, y1+y, w, h)

        if best_center and best_dist < 0.6 * ROI:
            tracker.update(best_center[0], best_center[1])
            used_measurement = True
            detection_bbox = best_box
            frames_with_detection += 1

            if best_dist < 40:
                accurate_detections += 1

            lost_frames = 0
        else:
            lost_frames += 1

        # PREDICTION BOX (RED)
        scale = pred_y / H
        bw = int(60 + scale * 120)
        bh = int(120 + scale * 220)

        cv2.rectangle(frame,
                      (pred_x-bw//2, pred_y-bh//2),
                      (pred_x+bw//2, pred_y+bh//2),
                      (0, 0, 255), 2)

        # DETECTION BOX (GREEN)
        if used_measurement and detection_bbox:
            dx, dy, dw, dh = detection_bbox
            cv2.rectangle(frame, (dx, dy), (dx+dw, dy+dh), (0,255,0), 2)

        if lost_frames > 90:
            tracking = False

    prev_frame = frame.copy()
    cv2.imshow("EKF Tracking", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# METRICS
print("Total Frames:", total_frames)
print("Detection Rate:", (frames_with_detection/total_frames)*100)
if frames_with_detection > 0:
    print("Detection Accuracy:", (accurate_detections/frames_with_detection)*100)
