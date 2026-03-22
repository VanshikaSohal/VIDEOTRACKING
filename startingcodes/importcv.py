import cv2
import numpy as np
# Global variables
clicked = False
init_x, init_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    global clicked, init_x, init_y
    if event == cv2.EVENT_LBUTTONDOWN:
        init_x, init_y = x, y
        clicked = True
        print(f"Object initialized at ({x}, {y})")

# Video capture
cap = cv2.VideoCapture("challenge.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

cv2.namedWindow("Kalman Only Tracking")
cv2.setMouseCallback("Kalman Only Tracking", mouse_callback)

# Kalman Filter setup
kalman = cv2.KalmanFilter(4, 2)

dt = 1
kalman.transitionMatrix = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
kalman.errorCovPost = np.eye(4, dtype=np.float32)

kalman_initialized = False

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if clicked and not kalman_initialized:
        kalman.statePost = np.array([
            [init_x],
            [init_y],
            [0],
            [0]
        ], dtype=np.float32)
        kalman_initialized = True

    if kalman_initialized:
        prediction = kalman.predict()
        x, y = int(prediction[0]), int(prediction[1])

        # Draw predicted position
        cv2.circle(frame, (x, y), 20, (255, 0, 0), 2)
        cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

        cv2.putText(frame, "Kalman Prediction",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

    else:
        cv2.putText(frame, "Click on object to initialize Kalman",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow("Kalman Only Tracking", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
