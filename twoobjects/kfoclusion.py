"""
SORT KF DUAL TRACKER - OCCLUSION HANDLING
- HOG added for occlusion recovery
- max_age extended for occlusion patience
- Occlusion state flagged via IoU overlap
"""

import cv2
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

# =======================
# ADVANCED KALMAN TRACKER
# =======================
class AdvancedKalmanTracker:
    next_id = 1

    def __init__(self, bbox):
        x,y,w,h=bbox
        self.id  = AdvancedKalmanTracker.next_id
        AdvancedKalmanTracker.next_id += 1
        self.trail       = deque(maxlen=50)
        self.is_occluded = False
        self.occluded_frames = 0

        self.kf=cv2.KalmanFilter(7,4,0,cv2.CV_32F)
        s=w*h; r=w/float(h) if h>0 else 1.0

        self.kf.transitionMatrix=np.array([
            [1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]],dtype=np.float32)

        self.kf.measurementMatrix=np.array([
            [1,0,0,0,0,0,0],[0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],[0,0,0,1,0,0,0]],dtype=np.float32)

        self.kf.processNoiseCov=np.eye(7,dtype=np.float32)*0.01
        self.kf.measurementNoiseCov=np.eye(4,dtype=np.float32)
        self.kf.measurementNoiseCov[0:2,0:2]*=1.0
        self.kf.measurementNoiseCov[2,2]*=10.0
        self.kf.measurementNoiseCov[3,3]*=10.0

        self.kf.statePost=np.array([[x],[y],[s],[r],[0],[0],[0]],dtype=np.float32)
        self.kf.errorCovPost=np.eye(7,dtype=np.float32)

        self.age=0; self.hits=1; self.hit_streak=1; self.time_since_update=0
        self.history=deque(maxlen=30); self.history.append((x,y))

    def predict(self):
        # Increase process noise during occlusion
        if self.is_occluded:
            self.kf.processNoiseCov=np.eye(7,dtype=np.float32)*0.05
        else:
            self.kf.processNoiseCov=np.eye(7,dtype=np.float32)*0.01

        if len(self.history)>=3: self._adjust_for_curved_path()
        pred=self.kf.predict()
        x=float(pred[0][0]); y=float(pred[1][0])
        s=max(100,float(pred[2][0]))
        r=max(0.2,min(5.0,float(pred[3][0])))
        self.age+=1; self.time_since_update+=1
        self.trail.append((int(x),int(y)))
        return self._to_bbox(x,y,s,r)

    def update(self, bbox):
        x,y,w,h=bbox
        s=w*h; r=w/float(h) if h>0 else 1.0
        meas=np.array([[np.float32(x)],[np.float32(y)],
                       [np.float32(s)],[np.float32(r)]])
        self.kf.correct(meas)
        self.hits+=1; self.hit_streak+=1; self.time_since_update=0
        self.history.append((x,y))
        self.is_occluded=False; self.occluded_frames=0

    def get_state(self):
        st=self.kf.statePost
        return self._to_bbox(float(st[0][0]),float(st[1][0]),
                             float(st[2][0]),float(st[3][0]))

    def center(self):
        st=self.kf.statePost
        return int(st[0][0]),int(st[1][0])

    def _to_bbox(self,x,y,s,r):
        w=np.sqrt(s*r); h=np.sqrt(s/r)
        return (int(x-w/2),int(y-h/2),int(w),int(h))

    def _adjust_for_curved_path(self):
        positions=list(self.history)[-5:]
        if len(positions)<3: return
        velocities=[(positions[i][0]-positions[i-1][0],
                     positions[i][1]-positions[i-1][1])
                    for i in range(1,len(positions))]
        if len(velocities)>=2:
            ax=np.mean([velocities[i][0]-velocities[i-1][0] for i in range(1,len(velocities))])
            ay=np.mean([velocities[i][1]-velocities[i-1][1] for i in range(1,len(velocities))])
            self.kf.statePost[4][0]+=ax*0.3
            self.kf.statePost[5][0]+=ay*0.3


# =======================
# SORT TRACKER
# =======================
class SORTTracker:
    def __init__(self, max_age=120, min_hits=1, iou_threshold=0.2, max_tracks=2):
        self.max_age=max_age; self.min_hits=min_hits
        self.iou_threshold=iou_threshold; self.max_tracks=max_tracks
        self.trackers=[]; self.frame_count=0

    def update(self, detections):
        self.frame_count+=1
        predictions=[]; to_del=[]
        for i,t in enumerate(self.trackers):
            predictions.append(t.predict())
            if t.time_since_update>self.max_age: to_del.append(i)
        for i in reversed(to_del):
            self.trackers.pop(i); predictions.pop(i)

        matched,unmatched_dets,_=self._associate(detections,predictions)
        for d_idx,t_idx in matched:
            self.trackers[t_idx].update(detections[d_idx])
        for d_idx in unmatched_dets:
            if len(self.trackers)<self.max_tracks:
                self.trackers.append(AdvancedKalmanTracker(detections[d_idx]))

        active=[]
        for t in self.trackers:
            if t.time_since_update<1 and t.hit_streak>=self.min_hits:
                active.append((*t.get_state(),t.id))
        return active

    def _associate(self, detections, predictions):
        if not predictions: return [],list(range(len(detections))),[]
        if not detections:  return [],[],list(range(len(predictions)))
        iou_mat=np.zeros((len(detections),len(predictions)))
        for d,det in enumerate(detections):
            for t,pred in enumerate(predictions):
                iou_mat[d,t]=self._iou(det,pred)
        row,col=linear_sum_assignment(-iou_mat)
        matched=[]; unmatched_dets=[]; unmatched_trks=[]
        for d,t in zip(row,col):
            if iou_mat[d,t]>=self.iou_threshold: matched.append((d,t))
            else: unmatched_dets.append(d); unmatched_trks.append(t)
        for d in range(len(detections)):
            if d not in row: unmatched_dets.append(d)
        for t in range(len(predictions)):
            if t not in col: unmatched_trks.append(t)
        return matched,unmatched_dets,unmatched_trks

    def _iou(self,b1,b2):
        x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
        xi1=max(x1,x2); yi1=max(y1,y2)
        xi2=min(x1+w1,x2+w2); yi2=min(y1+h1,y2+h2)
        inter=max(0,xi2-xi1)*max(0,yi2-yi1)
        union=w1*h1+w2*h2-inter
        return inter/union if union>0 else 0

    def flag_occlusions(self, iou_thresh=0.25):
        """Flag trackers whose predicted boxes heavily overlap"""
        trk_list=self.trackers
        for i in range(len(trk_list)):
            for j in range(i+1,len(trk_list)):
                b1=trk_list[i].get_state()
                b2=trk_list[j].get_state()
                if self._iou(b1,b2)>iou_thresh:
                    was_i=trk_list[i].is_occluded
                    trk_list[i].is_occluded=True
                    trk_list[j].is_occluded=True
                    if not was_i: trk_list[i].occluded_frames=0
                    trk_list[i].occluded_frames+=1
                    trk_list[j].occluded_frames+=1
                else:
                    # Only clear if not already set by another pair
                    pass


# =======================
# DETECTION FUNCTIONS
# =======================
def detect_motion_with_compensation(frame, prev_frame, optical_flow=None):
    if prev_frame is None or frame.shape!=prev_frame.shape: return [],None
    g1=cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
    g2=cv2.cvtColor(frame,     cv2.COLOR_BGR2GRAY)
    if optical_flow is None:
        p0=cv2.goodFeaturesToTrack(g1,maxCorners=100,qualityLevel=0.3,
                                   minDistance=7,blockSize=7)
        if p0 is not None:
            lk=dict(winSize=(15,15),maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))
            p1,st,_=cv2.calcOpticalFlowPyrLK(g1,g2,p0,None,**lk)
            if p1 is not None:
                gn=p1[st==1]; go=p0[st==1]
                if len(gn)>10:
                    optical_flow=(np.median(gn[:,0]-go[:,0]),
                                  np.median(gn[:,1]-go[:,1]))
    if optical_flow is not None:
        dx,dy=optical_flow
        M=np.float32([[1,0,-dx],[0,1,-dy]])
        g1=cv2.warpAffine(g1,M,(g1.shape[1],g1.shape[0]))

    g1=cv2.GaussianBlur(g1,(7,7),0); g2=cv2.GaussianBlur(g2,(7,7),0)
    diff=cv2.absdiff(g1,g2)
    _,th=cv2.threshold(diff,35,255,cv2.THRESH_BINARY)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    th=cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel,iterations=2)
    th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel)
    th=cv2.dilate(th,kernel,iterations=1)
    cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    H,W=frame.shape[:2]; boxes=[]
    for c in cnts:
        area=cv2.contourArea(c)
        if 150<area<30000:
            x,y,w,h=cv2.boundingRect(c)
            if x<50 or y<50 or x+w>W-50 or y+h>H-50: continue
            ar=float(h)/w if w>0 else 0
            if 1.5<ar<3.5 and y+h/2>H*0.3:
                boxes.append((x,y,w,h))
    return boxes, optical_flow

def detect_bg_improved(frame, bg):
    fg=bg.apply(frame,learningRate=0.005)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,kernel,iterations=2)
    fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,kernel)
    fg=cv2.medianBlur(fg,7)
    cnts,_=cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    H,W=frame.shape[:2]; boxes=[]
    for c in cnts:
        area=cv2.contourArea(c)
        if 3000<area<25000:
            x,y,w,h=cv2.boundingRect(c)
            if x<50 or y<50 or x+w>W-50 or y+h>H-50: continue
            ar=float(h)/w if w>0 else 0
            if 1.5<ar<3.5 and y+h/2>H*0.3:
                boxes.append((x,y,w,h))
    return boxes

def detect_hog(frame, hog):
    try:
        rects,_=hog.detectMultiScale(frame,winStride=(8,8),padding=(4,4),scale=1.05)
        return list(rects)
    except: return []

def non_max_suppression(boxes, thresh=0.3):
    if not boxes: return []
    boxes=np.array(boxes)
    x1=boxes[:,0]; y1=boxes[:,1]
    x2=boxes[:,0]+boxes[:,2]; y2=boxes[:,1]+boxes[:,3]
    area=(x2-x1)*(y2-y1); idxs=np.argsort(y2); pick=[]
    while len(idxs)>0:
        last=len(idxs)-1; i=idxs[last]; pick.append(i)
        xx1=np.maximum(x1[i],x1[idxs[:last]])
        yy1=np.maximum(y1[i],y1[idxs[:last]])
        xx2=np.minimum(x2[i],x2[idxs[:last]])
        yy2=np.minimum(y2[i],y2[idxs[:last]])
        overlap=(np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1))/area[idxs[:last]]
        idxs=np.delete(idxs,np.concatenate(([last],np.where(overlap>thresh)[0])))
    return boxes[pick].tolist()


COLORS        = [(0,0,255),(0,255,0),(255,128,0),(0,255,255),(255,0,255),(128,255,0)]
OCCLUDE_COLOR = (0,165,255)
MAX_OBJECTS   = 2


# =======================
# MAIN
# =======================
def main():
    cap=cv2.VideoCapture("Twopersonocllusion.mp4")
    if not cap.isOpened(): print("❌ Cannot open video"); return

    ret,frame=cap.read()
    prev_frame=frame.copy()
    H,W=frame.shape[:2]

    bg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=16,detectShadows=False)

    # HOG for occlusion recovery
    hog=cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Extended max_age=120, lower iou_threshold=0.2 for occlusion
    tracker=SORTTracker(max_age=120,min_hits=1,iou_threshold=0.2,max_tracks=MAX_OBJECTS)

    total_frames=0; optical_flow=None
    tracked_ids=set(); frame_w_tracks=0
    iou_scores=[]; total_dets=0
    false_positives=0; succ_matches=0
    total_preds=0; track_lifetimes={}
    occlusion_events=0

    print("="*60)
    print("SORT DUAL TRACKER - OCCLUSION VIDEO")
    print("="*60)
    print("🎮 Q/ESC=Quit | SPACE=Pause | R=Reset IDs\n")

    while True:
        ret,frame=cap.read()
        if not ret: break
        total_frames+=1
        total_preds+=len(tracker.trackers)

        motion_dets,optical_flow=detect_motion_with_compensation(frame,prev_frame,optical_flow)
        bg_dets=detect_bg_improved(frame,bg)

        # Add HOG detections when any tracker is occluded
        any_occluded=any(t.is_occluded for t in tracker.trackers)
        hog_dets=detect_hog(frame,hog) if any_occluded else []

        all_dets=non_max_suppression(motion_dets+bg_dets+hog_dets,thresh=0.4)
        total_dets+=len(all_dets)

        tracks=tracker.update(all_dets)

        # Flag occlusions after update
        prev_occ_count=sum(1 for t in tracker.trackers if t.is_occluded)
        tracker.flag_occlusions(iou_thresh=0.25)
        new_occ_count=sum(1 for t in tracker.trackers if t.is_occluded)
        if new_occ_count>prev_occ_count: occlusion_events+=1

        # Metrics
        for t in tracker.trackers:
            if t.time_since_update==0: succ_matches+=1
        if len(all_dets)>len(tracks): false_positives+=len(all_dets)-len(tracks)
        if tracks: frame_w_tracks+=1

        for track in tracks:
            tid=track[4]; tracked_ids.add(tid)
            track_lifetimes[tid]=track_lifetimes.get(tid,0)+1
            best_iou=max((tracker._iou(track[:4],d) for d in all_dets),default=0)
            if best_iou>0: iou_scores.append(best_iou)

        # ── Draw raw detections ──
        for (x,y,w,h) in all_dets:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,100),1)

        # ── Draw tracks ──
        for track in tracks:
            x,y,w,h,tid=track
            # Find tracker object
            t_obj=next((t for t in tracker.trackers if t.id==tid),None)
            is_occ=t_obj.is_occluded if t_obj else False

            color=OCCLUDE_COLOR if is_occ else COLORS[tid%len(COLORS)]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,3)

            label=f"ID:{tid} OCCLUDED" if is_occ else f"ID:{tid}"
            cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            cv2.circle(frame,(x+w//2,y+h//2),5,color,-1)

            if t_obj:
                pts=list(t_obj.trail)
                for i in range(1,len(pts)):
                    cv2.line(frame,pts[i-1],pts[i],color,2)

        # ── HUD ──
        det_rate=(total_dets/total_frames*100) if total_frames else 0
        track_rate=(frame_w_tracks/total_frames*100) if total_frames else 0
        match_rate=(succ_matches/total_preds*100) if total_preds else 0
        avg_iou=(np.mean(iou_scores)*100) if iou_scores else 0
        fp_rate=(false_positives/total_dets*100) if total_dets else 0
        occ_now=sum(1 for t in tracker.trackers if t.is_occluded)

        cv2.rectangle(frame,(0,0),(W,175),(40,40,40),-1)
        cv2.putText(frame,f"Frame:{total_frames} | Active:{len(tracks)}/{MAX_OBJECTS} | IDs:{len(tracked_ids)} | Occluded:{occ_now}",
                    (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2)
        cv2.putText(frame,f"Occlusion Events: {occlusion_events}",
                    (10,58),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,165,255),2)
        cv2.putText(frame,f"Tracking Rate:{track_rate:.1f}% | Match Rate:{match_rate:.1f}%",
                    (10,90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,200),2)
        cv2.putText(frame,f"IoU Quality:{avg_iou:.1f}% | FP Rate:{fp_rate:.1f}%",
                    (10,120),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,200,100),2)
        if optical_flow:
            dx,dy=optical_flow
            cv2.putText(frame,f"Camera Motion:{np.hypot(dx,dy):.1f}px",
                        (10,150),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,200,255),1)

        cv2.imshow("SORT Occlusion Tracker",frame)
        prev_frame=frame.copy()

        key=cv2.waitKey(30)&0xFF
        if key in (27,ord('q')): break
        elif key==ord(' '): cv2.waitKey(0)
        elif key==ord('r'):
            AdvancedKalmanTracker.next_id=1
            tracked_ids.clear(); track_lifetimes.clear(); occlusion_events=0
            print("🔄 Track IDs reset")

    cap.release()
    cv2.destroyAllWindows()

    track_rate=(frame_w_tracks/total_frames*100) if total_frames else 0
    match_rate=(succ_matches/total_preds*100) if total_preds else 0
    avg_iou=(np.mean(iou_scores)*100) if iou_scores else 0
    fp_rate=(false_positives/total_dets*100) if total_dets else 0
    overall=track_rate*0.4+match_rate*0.3+avg_iou*0.2+(100-fp_rate)*0.1

    print(f"\n{'='*60}\nFINAL REPORT\n{'='*60}")
    print(f"Total Frames:        {total_frames}")
    print(f"Unique Track IDs:    {len(tracked_ids)}")
    print(f"Occlusion Events:    {occlusion_events}")
    print(f"Tracking Rate:       {track_rate:.1f}%")
    print(f"Match Rate:          {match_rate:.1f}%")
    print(f"Avg IoU Quality:     {avg_iou:.1f}%")
    print(f"False Positive Rate: {fp_rate:.1f}%")
    print(f"Overall Score:       {overall:.1f}%")
    if track_lifetimes:
        print(f"Avg Track Lifetime:  {np.mean(list(track_lifetimes.values())):.1f} frames")
    print("="*60)

if __name__=="__main__":
    main()