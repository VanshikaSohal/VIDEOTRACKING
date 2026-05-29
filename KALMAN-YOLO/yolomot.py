

import os
import sys
import cv2
import numpy as np
import shutil
import glob
import subprocess
from pathlib import Path
from scipy.optimize import linear_sum_assignment#hungarian algorithm

MOT17_ROOT   = r"MOT17"        
OUTPUT_ROOT  = r"yolokfoutputs"    
TRACKEVAL_DIR= r"TrackEval"           

SEQS = [
    "MOT17-02-FRCNN",
    #"MOT17-04-FRCNN",
    #"MOT17-05-FRCNN",
    #"MOT17-09-FRCNN",
    #"MOT17-10-FRCNN",
    #"MOT17-11-FRCNN",
    #"MOT17-13-FRCNN",
]

# YOLOv8 detection settings
YOLO_CONF    = 0.25
YOLO_IOU     = 0.45

# KF tracker settings
MAX_AGE      = 30        # frames to keep a lost track alive
MIN_HITS     = 3         # min detections before track is confirmed
IOU_THRESHOLD= 0.3       # IoU threshold for Hungarian matching

# Video settings
VIDEO_FPS    = 10        # output video fps (lower = easier to watch)
VIDEO_W      = 1920
VIDEO_H      = 1080


# KALMAN FILTER TRACKER

def bbox_to_z(bbox):
    """[x1,y1,x2,y2] → [cx, cy, area, ratio]"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

def z_to_bbox(x, score=None):
    """KF state → [x1,y1,x2,y2] or [x1,y1,x2,y2,score]"""
    w = np.sqrt(abs(x[2] * x[3]))
    h = x[2] / (w + 1e-6)
    box = [x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]
    if score is None:
        return np.array(box)
    return np.array(box + [score])


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        # State transition matrix F (constant velocity model)
        self.kf_F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)

        # Measurement matrix H
        self.kf_H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)

        # Measurement noise R
        self.kf_R = np.eye(4, dtype=float)
        self.kf_R[2:, 2:] *= 10.

        # Process noise Q
        self.kf_Q = np.eye(7, dtype=float)
        self.kf_Q[4:, 4:] *= 0.01

        # State covariance P
        self.kf_P = np.eye(7, dtype=float)
        self.kf_P[4:, 4:] *= 1000.
        self.kf_P *= 10.

        # State vector x
        self.kf_x = np.zeros((7, 1), dtype=float)
        self.kf_x[:4] = bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self):
        if self.kf_x[6] + self.kf_x[2] <= 0:
            self.kf_x[6] = 0.
        # Predict step
        self.kf_x = self.kf_F @ self.kf_x
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(z_to_bbox(self.kf_x))
        return self.history[-1]

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        z = bbox_to_z(bbox)
        # Innovation
        y = z - self.kf_H @ self.kf_x
        S = self.kf_H @ self.kf_P @ self.kf_H.T + self.kf_R
        K = self.kf_P @ self.kf_H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(7) - K @ self.kf_H) @ self.kf_P

    def get_state(self):
        return z_to_bbox(self.kf_x)


def iou_batch(bb_test, bb_gt):
    """Compute IoU matrix between two sets of boxes [x1,y1,x2,y2]."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area_test = (bb_test[...,2]-bb_test[...,0]) * (bb_test[...,3]-bb_test[...,1])
    area_gt   = (bb_gt[...,2]-bb_gt[...,0])   * (bb_gt[...,3]-bb_gt[...,1])
    union = area_test + area_gt - inter
    return inter / (union + 1e-6)


class SORT:
    def __init__(self, max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD):
        self.max_age      = max_age
        self.min_hits     = min_hits
        self.iou_threshold= iou_threshold
        self.trackers     = []
        self.frame_count  = 0
        KalmanBoxTracker.count = 0   # reset IDs per sequence

    def update(self, dets):
        """
        dets: np.array shape (N,5) → [x1,y1,x2,y2,conf]
        returns: np.array shape (M,5) → [x1,y1,x2,y2,track_id]
        """
        self.frame_count += 1

        # Predict all existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t in range(len(self.trackers)):
           pos = self.trackers[t].predict().flatten()
           trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
           if np.any(np.isnan(pos)):
             to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Hungarian assignment
        matched, unmatched_dets, unmatched_trks = self._associate(dets, trks)

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        # Collect results
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d.flatten(), [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if ret:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def _associate(self, detections, trackers):
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))

        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched, unmatched_dets, unmatched_trks = [], [], []
        for d in range(len(detections)):
            if d not in row_ind:
                unmatched_dets.append(d)
        for t in range(len(trackers)):
            if t not in col_ind:
                unmatched_trks.append(t)
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] < self.iou_threshold:
                unmatched_dets.append(r)
                unmatched_trks.append(c)
            else:
                matched.append([r, c])

        return matched, unmatched_dets, unmatched_trks

PALETTE = [
    (255,  56,  56), (255, 157,  151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), (72,  249,  10), (146, 204,  23), (61,  219, 134),
    (26,  147, 52),  (0,  212, 187),  (44,  153, 168), (0,  194, 255),
    (52,   69, 147), (100,  115, 255),(0,   24, 236),  (132,  56, 255),
    (82,    0, 133), (203,  56, 255), (255, 149, 200), (255,  55, 199),
]

def get_color(track_id):
    return PALETTE[int(track_id) % len(PALETTE)]


def draw_tracks(frame, tracks, seq_name, frame_id, total_frames):
    h, w = frame.shape[:2]

    # ── Semi-transparent top bar for heading ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Sequence name
    cv2.putText(frame, f"Sequence: {seq_name}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Frame counter
    cv2.putText(frame, f"Frame: {frame_id} / {total_frames}  |  Tracks: {len(tracks)}",
                (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 1, cv2.LINE_AA)

    # Method label bottom-right
    cv2.putText(frame, "YOLOv8 + Kalman Filter",
                (w - 340, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)

    # ── Draw each track ──
    for trk in tracks:
        x1, y1, x2, y2, tid = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3]), int(trk[4])
        color = get_color(tid)

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ID label with background
        label = f"ID {tid}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame

def make_transition_slide(seq_name, w=VIDEO_W, h=VIDEO_H, n_frames=30):
    frames = []
    for i in range(n_frames):
        slide = np.zeros((h, w, 3), dtype=np.uint8)
        slide[:] = (30, 30, 30)
        # Thin horizontal accent line
        cv2.line(slide, (w//4, h//2 - 50), (3*w//4, h//2 - 50), (0, 200, 255), 2)
        cv2.line(slide, (w//4, h//2 + 50), (3*w//4, h//2 + 50), (0, 200, 255), 2)
        cv2.putText(slide, "YOLOv8 + Kalman Filter Tracker",
                    (w//2 - 380, h//2 - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(slide, seq_name,
                    (w//2 - 250, h//2 + 10), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(slide, "MOT17 Dataset",
                    (w//2 - 120, h//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 1, cv2.LINE_AA)
        frames.append(slide)
    return frames

def run_tracking_and_video():
    from ultralytics import YOLO

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    track_results_dir = os.path.join(OUTPUT_ROOT, "track_results")
    video_frames_dir  = os.path.join(OUTPUT_ROOT, "video_frames")
    os.makedirs(track_results_dir, exist_ok=True)
    os.makedirs(video_frames_dir, exist_ok=True)

    print("=" * 60)
    print("  Loading YOLOv8x model...")
    print("=" * 60)
    model = YOLO("yolov8x.pt")

    # Combined video writer
    video_path = os.path.join(OUTPUT_ROOT, "yolo_kf_tracking_all_sequences.mp4")
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (VIDEO_W, VIDEO_H))

    for seq in SEQS:
        img_dir  = os.path.join(MOT17_ROOT, seq, "img1")
        seq_info = os.path.join(MOT17_ROOT, seq, "seqinfo.ini")

        if not os.path.exists(img_dir):
            print(f"[WARN] {seq} not found, skipping.")
            continue

        # Read seqinfo for total frame count
        total_frames = 0
        if os.path.exists(seq_info):
            with open(seq_info) as f:
                for line in f:
                    if line.startswith("seqLength"):
                        total_frames = int(line.strip().split("=")[1])

        frames = sorted(os.listdir(img_dir))
        if total_frames == 0:
            total_frames = len(frames)

        print(f"\n{'='*60}")
        print(f"  Processing: {seq}  ({len(frames)} frames)")
        print(f"{'='*60}")

        # Write transition slide
        for slide in make_transition_slide(seq, VIDEO_W, VIDEO_H):
            writer.write(slide)

        # Tracking result file (MOTChallenge format)
        result_file = os.path.join(track_results_dir, f"{seq}.txt")
        tracker = SORT()
        KalmanBoxTracker.count = 0

        with open(result_file, "w") as rf:
            for fi, frame_file in enumerate(frames):
                frame_id = int(frame_file.replace(".jpg","").replace(".png",""))
                img_path = os.path.join(img_dir, frame_file)

                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                # Resize to standard resolution for video
                frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))

                # ── YOLOv8 detection ──
                results = model(img_path, conf=YOLO_CONF, iou=YOLO_IOU,
                                classes=[0], verbose=False)

                dets = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        dets.append([x1, y1, x2, y2, conf])

                dets_np = np.array(dets) if dets else np.empty((0, 5))

                # ── Kalman Filter update ──
                tracks = tracker.update(dets_np)

                # ── Write MOTChallenge result ──
                for trk in tracks:
                    x1, y1, x2, y2, tid = trk
                    w_box = x2 - x1
                    h_box = y2 - y1
                    rf.write(f"{frame_id},{int(tid)},{x1:.2f},{y1:.2f},{w_box:.2f},{h_box:.2f},1,-1,-1,-1\n")

                # ── Draw & write video frame ──
                frame = draw_tracks(frame, tracks, seq, frame_id, total_frames)
                writer.write(frame)

                if (fi + 1) % 100 == 0:
                    print(f"  [{seq}] {fi+1}/{len(frames)} frames done...")

        print(f"  [{seq}] Tracking complete. Results saved.")

    writer.release()
    print(f"\n Combined video saved → {video_path}")
    return track_results_dir


def setup_trackeval(track_results_dir):
    print("\n" + "="*60)
    print("  Setting up TrackEval...")
    print("="*60)

    # Clone TrackEval if not present
    if not os.path.exists(TRACKEVAL_DIR):
        print("  Cloning TrackEval repo...")
        subprocess.run(["git", "clone",
                        "https://github.com/JonathonLuiten/TrackEval.git",
                        TRACKEVAL_DIR], check=True)

    # Install TrackEval deps
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "trackeval", "-q"], check=False)

    TRACKER_NAME = "YOLO_KF"

    for seq in SEQS:
        # GT
        dst_gt  = os.path.join(TRACKEVAL_DIR, "data", "gt", "mot_challenge",
                               "MOT17-train", seq, "gt")
        dst_ini = os.path.join(TRACKEVAL_DIR, "data", "gt", "mot_challenge",
                               "MOT17-train", seq)
        # Tracker
        dst_trk = os.path.join(TRACKEVAL_DIR, "data", "trackers", "mot_challenge",
                               "MOT17-train", TRACKER_NAME, "data")

        os.makedirs(dst_gt,  exist_ok=True)
        os.makedirs(dst_trk, exist_ok=True)

        src_gt    = os.path.join(MOT17_ROOT, seq, "gt", "gt.txt")
        src_ini   = os.path.join(MOT17_ROOT, seq, "seqinfo.ini")
        src_trk   = os.path.join(track_results_dir, f"{seq}.txt")

        if os.path.exists(src_gt):
            shutil.copy(src_gt,  dst_gt)
        if os.path.exists(src_ini):
            shutil.copy(src_ini, dst_ini)
        if os.path.exists(src_trk):
            shutil.copy(src_trk, dst_trk)

    # Seqmap
    seqmap_dir = os.path.join(TRACKEVAL_DIR, "data", "gt", "mot_challenge", "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    with open(os.path.join(seqmap_dir, "MOT17-train.txt"), "w") as f:
        f.write("name\n" + "\n".join(SEQS))

    # Fix deprecated numpy types in TrackEval source
    for path in glob.glob(os.path.join(TRACKEVAL_DIR, "**", "*.py"), recursive=True):
        with open(path, "r", errors="ignore") as f:
            code = f.read()
        code = (code.replace("np.float,",  "float,")
                    .replace("np.float)",  "float)")
                    .replace("np.int,",    "int,")
                    .replace("np.int)",    "int)")
                    .replace("np.bool,",   "bool,")
                    .replace("np.bool)",   "bool)"))
        with open(path, "w") as f:
            f.write(code)

    return TRACKER_NAME


def run_evaluation(tracker_name):
    print("\n" + "="*60)
    print("  Running TrackEval — HOTA / CLEAR / Identity")
    print("="*60)

    script = os.path.join(TRACKEVAL_DIR, "scripts", "run_mot_challenge.py")
    cmd = [
        sys.executable, script,
        "--BENCHMARK",        "MOT17",
        "--SPLIT_TO_EVAL",    "train",
        "--TRACKERS_TO_EVAL", tracker_name,
        "--METRICS",          "HOTA", "CLEAR", "Identity",
        "--GT_FOLDER",        os.path.join(TRACKEVAL_DIR, "data", "gt", "mot_challenge"),
        "--TRACKERS_FOLDER",  os.path.join(TRACKEVAL_DIR, "data", "trackers", "mot_challenge"),
        "--USE_PARALLEL",     "False",
        "--NUM_PARALLEL_CORES", "1",
    ]
    subprocess.run(cmd, check=True)

    # Print results summary
    results_path = os.path.join(
        TRACKEVAL_DIR, "data", "trackers", "mot_challenge",
        "MOT17-train", tracker_name, "pedestrian_summary.txt"
    )
    if os.path.exists(results_path):
        print("\n" + "="*60)
        print("  FINAL SCORES")
        print("="*60)
        with open(results_path) as f:
            print(f.read())
    else:
        print(f"[INFO] Results at: {TRACKEVAL_DIR}/data/trackers/mot_challenge/MOT17-train/{tracker_name}/")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  YOLOv8 + Kalman Filter — MOT17 Tracker")
    print("="*60)

    # Step 1: Run tracking + generate video
    track_results_dir = run_tracking_and_video()

    # Step 2: Setup TrackEval
    tracker_name = setup_trackeval(track_results_dir)

    # Step 3: Evaluate
    run_evaluation(tracker_name)

    print("\n ALL DONE!")
    print(f"   Video  → {OUTPUT_ROOT}/yolo_kf_tracking_all_sequences.mp4")
    print(f"    Scores → {TRACKEVAL_DIR}/data/trackers/mot_challenge/MOT17-train/YOLO_KF/")