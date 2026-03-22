"""
EKF DUAL TRACKER - OCCLUSION HANDLING
- HOG added specifically for occlusion recovery
- Occlusion detection via bbox overlap
- Extended lost frame patience
- Wider ROI during occlusion
"""

import cv2
import numpy as np
from collections import deque
from filterpy.kalman import ExtendedKalmanFilter as EKF


def hx(x):
    return np.array([x[0], x[1]])

def H_jacobian(x):
    return np.array([[1,0,0,0],[0,1,0,0]])

class EKFTracker:
    def __init__(self, x, y, tracker_id, dt=1.0):
        self.id          = tracker_id
        self.lost_frames = 0
        self.trail       = deque(maxlen=50)
        self.is_occluded = False
        self.occluded_frames = 0
        self.last_known_pos  = (x, y)
        self.last_known_vel  = (0.0, 0.0)

        self.ekf = EKF(dim_x=4, dim_z=2)
        self.ekf.x = np.array([x, y, 0, 0], dtype=float)
        self.ekf.F = np.array([
            [1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.ekf.H  = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.ekf.P *= 10.0
        self.ekf.R  = np.eye(2) * 5.0
        self.ekf.Q  = np.eye(4) * 0.05
        self.prev_positions = [(x, y)] * 3

    def predict(self):
        # During occlusion, increase process noise (more uncertainty)
        if self.is_occluded:
            self.ekf.Q = np.eye(4) * 0.15
        else:
            self.ekf.Q = np.eye(4) * 0.05

        self.ekf.predict()
        px, py = int(self.ekf.x[0]), int(self.ekf.x[1])
        self.prev_positions.append((px, py))
        if len(self.prev_positions) > 3: self.prev_positions.pop(0)
        sx = int(np.mean([p[0] for p in self.prev_positions]))
        sy = int(np.mean([p[1] for p in self.prev_positions]))
        self.trail.append((sx, sy))
        return sx, sy

    def update(self, x, y):
        self.ekf.update(np.array([x, y], dtype=float),
                        HJacobian=H_jacobian, Hx=hx)
        self.last_known_pos = (x, y)
        self.last_known_vel = (float(self.ekf.x[2]), float(self.ekf.x[3]))
        self.is_occluded    = False
        self.occluded_frames = 0

    def get_velocity(self):
        return self.ekf.x[2], self.ekf.x[3]

    def center(self):
        return int(self.ekf.x[0]), int(self.ekf.x[1])


def compute_iou(b1, b2):
    """IoU between two (cx,cy,w,h) boxes"""
    x1,y1,w1,h1 = b1[0]-b1[2]//2, b1[1]-b1[3]//2, b1[2], b1[3]
    x2,y2,w2,h2 = b2[0]-b2[2]//2, b2[1]-b2[3]//2, b2[2], b2[3]
    xi1=max(x1,x2); yi1=max(y1,y2)
    xi2=min(x1+w1,x2+w2); yi2=min(y1+h1,y2+h2)
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    union=w1*h1+w2*h2-inter
    return inter/union if union > 0 else 0

def detect_occlusion(trackers, predictions, H, iou_thresh=0.3):
   
    occluded = [False] * len(trackers)
    if len(trackers) < 2:
        return occluded

    # Build prediction boxes [cx, cy, w, h]
    pred_boxes = []
    for t_idx, (px, py) in enumerate(predictions):
        scale = py / H
        bw = int(65 + scale*110)
        bh = int(130 + scale*200)
        pred_boxes.append((px, py, bw, bh))

    for i in range(len(trackers)):
        for j in range(i+1, len(trackers)):
            iou = compute_iou(pred_boxes[i], pred_boxes[j])
            if iou > iou_thresh:
                occluded[i] = True
                occluded[j] = True

    return occluded


def detect_motion(frame, prev_frame):
    if prev_frame is None or frame.shape != prev_frame.shape: return []
    g1 = cv2.GaussianBlur(cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY),(21,21),0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(frame,     cv2.COLOR_BGR2GRAY),(21,21),0)
    diff = cv2.absdiff(g1, g2)
    _, th = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
    th = cv2.dilate(th, kernel, iterations=5)
    th = cv2.erode(th, kernel, iterations=3)
    cnts,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in cnts:
        area=cv2.contourArea(c)
        if 400<area<50000:
            x,y,w,h=cv2.boundingRect(c)
            ar=h/float(w) if w>0 else 0
            if 1.1<ar<5.5 and h>18 and w>12:
                pw,ph=int(w*0.6),int(h*0.4)
                boxes.append((max(0,x-pw),max(0,y-ph),w+2*pw,h+2*ph))
    return boxes

def detect_bg(frame, bg):
    fg=bg.apply(frame,learningRate=0.012)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,kernel)
    fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,kernel)
    cnts,_=cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in cnts:
        area=cv2.contourArea(c)
        if 500<area<45000:
            x,y,w,h=cv2.boundingRect(c)
            ar=h/float(w) if w>0 else 0
            if 1.1<ar<5.5 and h>22 and w>12:
                pw,ph=int(w*0.6),int(h*0.4)
                boxes.append((max(0,x-pw),max(0,y-ph),w+2*pw,h+2*ph))
    return boxes

def detect_hog(frame, hog):
    try:
        rects,_ = hog.detectMultiScale(frame, winStride=(8,8), padding=(4,4), scale=1.05)
        return list(rects)
    except:
        return []

def combine_detections(boxes, max_dist=180):
    if not boxes: return []
    merged, used = [], set()
    for i,(x1,y1,w1,h1) in enumerate(boxes):
        if i in used: continue
        cx1,cy1=x1+w1//2,y1+h1//2
        group=[(x1,y1,w1,h1)]
        for j,(x2,y2,w2,h2) in enumerate(boxes[i+1:],i+1):
            if j in used: continue
            if np.hypot(cx1-(x2+w2//2),cy1-(y2+h2//2))<max_dist:
                group.append((x2,y2,w2,h2)); used.add(j)
        ax=[b[0] for b in group]; ay=[b[1] for b in group]
        ax2=[b[0]+b[2] for b in group]; ay2=[b[1]+b[3] for b in group]
        merged.append((min(ax),min(ay),max(ax2)-min(ax),max(ay2)-min(ay)))
        used.add(i)
    return merged

def assign_detections_to_trackers(trackers, detections, max_dist=300):
    assignments, assigned = [], set()
    for t_idx,t in enumerate(trackers):
        tcx,tcy=t.center()
        best_dist,best_d=max_dist,-1
        for d_idx,(x,y,w,h) in enumerate(detections):
            if d_idx in assigned: continue
            d=np.hypot(x+w//2-tcx,y+h//2-tcy)
            if d<best_dist: best_dist,best_d=d,d_idx
        if best_d>=0:
            assignments.append((t_idx,detections[best_d])); assigned.add(best_d)
    return assignments


PRED_COLORS   = [(0,0,255),(255,128,0)]
DET_COLORS    = [(0,255,0),(0,255,255)]
OCCLUDE_COLOR = (0,165,255)   # Orange = occluded state
MAX_OBJECTS   = 2
MAX_LOST      = 120           # Much longer for occlusion patience
MIN_CONF      = 2


print("EKF DUAL TRACKER - OCCLUSION VIDEO")

VIDEO_FILE = "Twopersonocllusion.mp4"
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f" Cannot open '{VIDEO_FILE}'"); exit()

ret, frame = cap.read()
if not ret: print(" Cannot read first frame"); exit()

prev_frame = frame.copy()
H, W = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"📹 {W}x{H} @ {fps:.1f} FPS  |  🎮 ESC=Quit SPACE=Pause R=Reset\n")

bg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=18,detectShadows=False)

# HOG for occlusion recovery
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

trackers    = []
next_id     = 0
consecutive = 0

total_frames          = 0
frames_with_detection = [0]*MAX_OBJECTS
accurate_detections   = [0]*MAX_OBJECTS
occlusion_events      = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    total_frames += 1

    all_dets = combine_detections(
        detect_motion(frame,prev_frame)+detect_bg(frame,bg), max_dist=200)
    all_dets = [b for b in all_dets if b[2]>20 and b[3]>45]

    if len(trackers) < MAX_OBJECTS and all_dets:
        consecutive += 1
        if consecutive >= MIN_CONF:
            existing = [t.center() for t in trackers]
            for (x,y,w,h) in sorted(all_dets,key=lambda b:b[2]*b[3],reverse=True):
                if len(trackers)>=MAX_OBJECTS: break
                cx,cy=x+w//2,y+h//2
                if not any(np.hypot(cx-ex,cy-ey)<80 for ex,ey in existing):
                    trackers.append(EKFTracker(cx,cy,next_id,
                                               dt=1.0/fps if fps>0 else 0.033))
                    existing.append((cx,cy))
                    print(f" Frame {total_frames}: EKF Obj {next_id} initialised")
                    next_id+=1
    else:
        if not all_dets: consecutive=0

    if not trackers:
        prev_frame=frame.copy()
        cv2.imshow("EKF Occlusion Tracker",frame)
        if cv2.waitKey(30)&0xFF==27: break
        continue

    predictions = [t.predict() for t in trackers]

    occluded_flags = detect_occlusion(trackers, predictions, H, iou_thresh=0.25)
    for t_idx,t in enumerate(trackers):
        was_occluded = t.is_occluded
        t.is_occluded = occluded_flags[t_idx]
        if t.is_occluded:
            t.occluded_frames += 1
            if not was_occluded: occlusion_events += 1

    all_roi_dets = []
    for t_idx,(pred_x,pred_y) in enumerate(predictions):
        t = trackers[t_idx]
        # Wider ROI when occluded
        base_roi = 350 if t.is_occluded else 280
        ROI = min(base_roi + t.lost_frames*25, 700)
        x1=max(0,pred_x-ROI); y1=max(0,pred_y-ROI)
        x2=min(W,pred_x+ROI); y2=min(H,pred_y+ROI)
        roi=frame[y1:y2,x1:x2]; prev_roi=prev_frame[y1:y2,x1:x2]

        local = combine_detections(
            detect_motion(roi,prev_roi)+detect_bg(roi,bg), max_dist=200)

        if t.is_occluded or t.lost_frames > 10:
            hog_dets = detect_hog(roi, hog)
            local += hog_dets

        all_roi_dets.extend([(x1+x,y1+y,w,h) for (x,y,w,h) in local])

    all_roi_dets = combine_detections(all_roi_dets)
    assignments  = assign_detections_to_trackers(trackers, all_roi_dets)
    assigned_ids = set()

    for (t_idx,(x,y,w,h)) in assignments:
        t = trackers[t_idx]
        pred_x,pred_y = predictions[t_idx]
        cx,cy = x+w//2, y+h//2
        dist = np.hypot(cx-pred_x, cy-pred_y)
        # More lenient threshold during occlusion
        factor = 0.85 if t.is_occluded else 0.65
        dist_threshold = factor * min(280+t.lost_frames*25, 700)

        if dist < dist_threshold:
            t.update(cx, cy)
            t.lost_frames = 0
            assigned_ids.add(t_idx)
            obj_idx = t_idx % MAX_OBJECTS
            frames_with_detection[obj_idx] += 1
            if dist < 45: accurate_detections[obj_idx] += 1
            cv2.rectangle(frame,(x,y),(x+w,y+h),DET_COLORS[obj_idx],3)
            cv2.putText(frame,f"EKF Obj {t.id} DETECT",
                        (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,DET_COLORS[obj_idx],2)

    for t_idx,t in enumerate(trackers):
        pred_x,pred_y=predictions[t_idx]
        obj_idx=t_idx%MAX_OBJECTS

        if t_idx not in assigned_ids:
            t.lost_frames += 1

        vx,vy=t.get_velocity()
        vel=np.sqrt(vx**2+vy**2)
        scale=pred_y/H
        bw=max(50,min(int(65+scale*110+vel*5),220))
        bh=max(100,min(int(130+scale*200+vel*8),380))

        if t.is_occluded:
            box_color = OCCLUDE_COLOR
            label = f"EKF Obj {t.id} OCCLUDED ({t.occluded_frames}f)"
        elif t_idx not in assigned_ids:
            box_color = PRED_COLORS[obj_idx]
            label = f"EKF Obj {t.id} PRED ({t.lost_frames}f)"
        else:
            box_color = PRED_COLORS[obj_idx]
            label = f"EKF Obj {t.id}"

        cv2.rectangle(frame,
                      (pred_x-bw//2,pred_y-bh//2),
                      (pred_x+bw//2,pred_y+bh//2), box_color,3)
        cv2.putText(frame,label,
                    (pred_x-bw//2,pred_y-bh//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,2)

        pts=list(t.trail)
        for i in range(1,len(pts)):
            cv2.line(frame,pts[i-1],pts[i],box_color,2)

    before=len(trackers)
    trackers=[t for t in trackers if t.lost_frames<=MAX_LOST]
    if len(trackers)<before:
        print(f" Frame {total_frames}: Lost tracker"); consecutive=0

    cv2.rectangle(frame,(0,0),(W,90),(40,40,40),-1)
    occ_now = sum(1 for t in trackers if t.is_occluded)
    cv2.putText(frame,
                f"Frame:{total_frames} | EKF {len(trackers)}/{MAX_OBJECTS} | Occluded:{occ_now} | Events:{occlusion_events}",
                (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
    cv2.putText(frame,"Orange=Occluded  Red/Orange=Predicting  Green/Yellow=Detected",
                (10,65),cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,180),1)

    prev_frame=frame.copy()
    cv2.imshow("EKF Occlusion Tracker",frame)
    key=cv2.waitKey(30)&0xFF
    if key==27: break
    elif key==ord(' '): cv2.waitKey(0)
    elif key==ord('r'):
        trackers=[]; consecutive=0; next_id=0; occlusion_events=0
        bg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=18,detectShadows=False)
        print(" RESET")

cap.release()
cv2.destroyAllWindows()

print(f"FINAL STATISTICS\n")
print(f"Total Frames:      {total_frames}")
print(f"Occlusion Events:  {occlusion_events}")
for i in range(MAX_OBJECTS):
    print(f"\nObject {i+1}:")
    print(f"  Detection Rate:     {(frames_with_detection[i]/total_frames)*100:.1f}%")
    if frames_with_detection[i]>0:
        print(f"  Detection Accuracy: {(accurate_detections[i]/frames_with_detection[i])*100:.1f}%")