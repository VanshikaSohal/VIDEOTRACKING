size how to define for area,height


# Keys
    key = cv2.waitKey(80) & 0xFF
    # this line is used for speed changing larger value means slower speed


self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01#prediction stability
lower value more stable prediction 

self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0#detection trust
lower value more trust

 self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.5#convergence speed
 lower value more confidence in initial state

self.prev_x = [x] * 2#smoothing buffers
self.prev_y = [y] * 2#small buffer for faster response

if dist <180 and dist < best_dist:
#tighter search area

✅ Red box leads green box = Prediction is working
✅ Red box steady, not jumpy = Good noise tuning
✅ Green box follows red smoothly = Good measurement trust
❌ Red box jumps around = processNoiseCov too high
❌ Red box ignores green = measurementNoiseCov too high
❌ Red box lags behind green = Buffer too large


        if area < 1500 or area > 25000:
        for area of bounding box around the person 

#sort/deepsort trackers of kf 
#lower threshold
#track width and height in kf state vector
#allow kf prediction for 5-10missed frames
#scaling factor in state vector of kf 
#curved paths
#moving camera
