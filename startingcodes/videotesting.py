import cv2

# Load video
cap = cv2.VideoCapture("challenge.mp4")
#cap is used to capture video from the specified file
#videocapture is a function in opencv that allows us to read video files
if not cap.isOpened():
    #isOpened() checks if the video file was opened successfully
    print("Error: Video could not be opened!")
else:
    print("Video loaded successfully!")
#this is a loop to read and display video frames
    while True:
        ret, frame = cap.read()
#ret is a boolean that indicates if the frame was read successfully
# frame is the actual frame read from the video    
        if not ret:
            break
            
        cv2.imshow('Test Video', frame)
#imshow displays the frame in a window named 'Test Video'
#name of the window can be anything you choose
        # Press 'q' to close
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
#waitkey(30) waits for 30 milliseconds for a key event
#0xFF is used to get the last 8 bits of the key code
#ord('q') gets the ASCII value of 'q'
#& is a bitwise AND operation to compare the key code with 'q'
#30 milliseconds is the delay between frames, adjust as needed
cap.release()
#release() releases the video capture object
cv2.destroyAllWindows()
#destroyAllWindows() closes all OpenCV windows