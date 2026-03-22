import cv2
import math

cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Error: Video could not be opened!")
    exit()
else:
    print("Video loaded successfully!")

frame_count = 0
paused=False

while True:
    if not paused:
     ret, frame = cap.read()
     frame_count += 1

    if not ret:
        print(f"Video ended at frame {frame_count}")
        break
   
    original_frame = frame.copy()
  
    # preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # find contours
    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    #RETR_EXTERNAL retrieves only the extreme outer contours
    #CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
    ball_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by size 
        if 20 < area < 2000:
            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            # Circularity check (≈1 means perfect circle)
            circularity = 4 * math.pi * area / (perimeter * perimeter)

            if circularity > 0.3:   # ball-like
                if area > max_area:
                    max_area = area
                    ball_contour = contour

    #  BALL DETECTED 
    if ball_contour is not None:
        x, y, w, h = cv2.boundingRect(ball_contour)
        center_x = x + w // 2
        center_y = y + h // 2
        radius = max(w, h) // 2

        cv2.circle(original_frame, (center_x, center_y),
                   radius, (0, 255, 0), 2)
        cv2.circle(original_frame, (center_x, center_y),
                   5, (0, 0, 255), -1)

        cv2.putText(original_frame,
                    f"Ball: ({center_x}, {center_y})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

        print(f"Frame {frame_count}: Ball at ({center_x}, {center_y})")

    # BALL NOT DETECTED =
    else:
        cv2.putText(original_frame, "Ball not detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        print(f"Frame {frame_count}: Ball not detected")

    if paused:
        cv2.putText(original_frame, "PAUSED - Press SPACE", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ================= DISPLAY =================
    cv2.imshow("Ball Detection (No Kalman)", original_frame)

    # KEY HANDLING
    key = cv2.waitKey(30) & 0xFF
    
    if key == ord('q'):
        print("\nStopped by user")
        break
    elif key == ord(' '):  # SPACEBAR
        paused = not paused
        if paused:
            print("PAUSED")
        else:
            print("RESUMED")

     
# CLEANUP
cap.release()
cv2.destroyAllWindows()
print(f"\nProcessing complete! Total frames: {frame_count}")
