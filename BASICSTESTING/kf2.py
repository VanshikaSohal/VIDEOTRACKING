import cv2
import numpy as np
from ultralytics import YOLO  # YOLOv8 package

# Video capture
cap = cv2.VideoCapture("challenge.mp4")

if not cap.isOpened():
    print("Error: Video could not be opened!")
    exit()
else:
    print("Video loaded successfully!")
    print("GREEN circle = Raw detection (jumpy)")
    print("BLUE circle = Kalman filtered (smooth)")
    print("Press Q to quit, SPACE to pause\n")

# YOLO Model
# Use pretrained YOLOv8n (nano) for speed
model = YOLO("yolov8n.pt") 

# Kalman Filter setup
kalman = cv2.KalmanFilter(4, 2)  # 4 states: x,y,vx,vy ; 2 measurements: x,y
dt = 1

kalman.transitionMatrix = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10
kalman.errorCovPost = np.eye(4, dtype=np.float32)

kalman_initialized = False
frame_count = 0
paused = False

# Main loop
while True:
    if not paused:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print(f"Video ended at frame {frame_count}")
            break

    original_frame = frame.copy()

    
    # YOLO detection
    results = model(frame)[0]  # Get predictions for current frame

    car_contour = None
    detected_x = detected_y = None

    # Iterate over detected objects
    for r in results.boxes:
        cls = int(r.cls[0])
        # YOLO class 2 = car (COCO dataset)
        if cls == 2:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detected_x = (x1 + x2) // 2
            detected_y = (y1 + y2) // 2
            w = x2 - x1
            h = y2 - y1
            radius = max(w, h) // 2
            break  # Track only first detected car (or remove break for multi-car)
    
    kalman_x = kalman_y = None

    # Kalman tracking
    if detected_x is not None and detected_y is not None:
        if not kalman_initialized:
            kalman.statePre = np.array([[detected_x],
                                        [detected_y],
                                        [0],
                                        [0]], dtype=np.float32)
            kalman.statePost = kalman.statePre.copy()
            kalman_initialized = True
            print(f"Kalman initialized at ({detected_x}, {detected_y})")

        prediction = kalman.predict()
        kalman.correct(np.array([[detected_x], [detected_y]], dtype=np.float32))
        kalman_x = int(kalman.statePost[0])
        kalman_y = int(kalman.statePost[1])

        # Draw raw detection (GREEN)
        cv2.circle(original_frame, (detected_x, detected_y), radius, (0, 255, 0), 2)
        cv2.circle(original_frame, (detected_x, detected_y), 3, (0, 255, 0), -1)

        # Draw Kalman filtered (BLUE)
        cv2.circle(original_frame, (kalman_x, kalman_y), radius, (255, 0, 0), 2)
        cv2.circle(original_frame, (kalman_x, kalman_y), 5, (255, 0, 0), -1)

        cv2.putText(original_frame, f"Raw: ({detected_x}, {detected_y})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(original_frame, f"Kalman: ({kalman_x}, {kalman_y})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        print(f"Frame {frame_count}: Raw=({detected_x},{detected_y}) | Kalman=({kalman_x},{kalman_y})")

    elif kalman_initialized:
        prediction = kalman.predict()
        kalman_x = int(prediction[0])
        kalman_y = int(prediction[1])

        cv2.circle(original_frame, (kalman_x, kalman_y), 20, (255, 0, 0), 2)
        cv2.circle(original_frame, (kalman_x, kalman_y), 5, (255, 0, 0), -1)

        cv2.putText(original_frame, "Detection lost - Using Kalman prediction",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(original_frame, f"Predicted: ({kalman_x}, {kalman_y})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        print(f"Frame {frame_count}: Detection lost | Kalman prediction=({kalman_x},{kalman_y})")

    else:
        cv2.putText(original_frame, "Searching for car...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Legend
    cv2.putText(original_frame, "GREEN = Raw detection",
                (10, original_frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(original_frame, "BLUE = Kalman filtered",
                (10, original_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if paused:
        cv2.putText(original_frame, "PAUSED - Press SPACE",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display
    cv2.imshow("YOLO + Kalman Car Tracking", original_frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        print("\nStopped by user")
        break
    elif key == ord(' '):
        paused = not paused
        print("PAUSED" if paused else "RESUMED")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"\nProcessing complete! Total frames: {frame_count}")
