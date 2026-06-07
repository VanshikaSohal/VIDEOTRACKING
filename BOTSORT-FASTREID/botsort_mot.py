import os
import sys
import cv2
import numpy as np
import shutil
import glob
import subprocess
from pathlib import Path

# Add BoT-SORT to path
BOTSORT_ROOT = r"C:\Users\vansh\VIDEOTRACKING\BOTSORT-FASTREID\BoT-SORT"
sys.path.insert(0, BOTSORT_ROOT)

MOT17_ROOT    = r"C:\Users\vansh\VANSHIKASOHAL\DiffMOT-main\datasets\MOT17"
OUTPUT_ROOT   = r"C:\Users\vansh\VIDEOTRACKING\BOTSORT-FASTREID\BOTSORT-OUTPUT"
TRACKEVAL_DIR = r"C:\Users\vansh\VIDEOTRACKING\BOTSORT-FASTREID\TrackEval"

SEQS = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]

VIDEO_FPS = 10
VIDEO_W   = 1920
VIDEO_H   = 1080

PALETTE = [
    (255,56,56),(255,157,151),(255,112,31),(255,178,29),
    (207,210,49),(72,249,10),(146,204,23),(61,219,134),
    (26,147,52),(0,212,187),(44,153,168),(0,194,255),
    (52,69,147),(100,115,255),(0,24,236),(132,56,255),
    (82,0,133),(203,56,255),(255,149,200),(255,55,199),
]

def get_color(track_id):
    return PALETTE[int(track_id) % len(PALETTE)]

def draw_tracks(frame, tracks, seq_name, frame_id, total_frames):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,70), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Sequence: {seq_name}",
                (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_id} / {total_frames}  |  Tracks: {len(tracks)}",
                (20,58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,220,255), 1, cv2.LINE_AA)
    cv2.putText(frame, "YOLOv8n det.txt + BoTSORT (FastReID SBS-S50)",
                (w-620, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,128), 2, cv2.LINE_AA)
    for trk in tracks:
        x1,y1,x2,y2,tid = int(trk[0]),int(trk[1]),int(trk[2]),int(trk[3]),int(trk[4])
        color = get_color(tid)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = f"ID {tid}"
        (lw,lh),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1,y1-lh-8), (x1+lw+6,y1), color, -1)
        cv2.putText(frame, label, (x1+3,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return frame

def load_detections(det_file):
    """Load pre-computed YOLOv8n detections from det.txt"""
    det_dict = {}
    with open(det_file) as f:
        for line in f:
            parts = line.strip().split(',')
            fid  = int(parts[0])
            x1   = float(parts[2])
            y1   = float(parts[3])
            w    = float(parts[4])
            h    = float(parts[5])
            conf = float(parts[6])
            if fid not in det_dict:
                det_dict[fid] = []
            det_dict[fid].append([x1, y1, x1+w, y1+h, conf, 0])
    return det_dict

def run_tracking_and_video():
    from tracker.bot_sort import BoTSORT

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    track_results_dir = os.path.join(OUTPUT_ROOT, "track_results")
    os.makedirs(track_results_dir, exist_ok=True)

    print("="*60)
    print("  YOLOv8n det.txt + BoTSORT (FastReID SBS-S50) — MOT17")
    print("  Using pre-computed detections — NO live YOLO!")
    print("="*60)

    video_path = os.path.join(OUTPUT_ROOT, "botsort_fastreid_tracking.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (VIDEO_W, VIDEO_H))

    class Args:
        track_high_thresh   = 0.3
        track_low_thresh    = 0.05
        new_track_thresh    = 0.25
        track_buffer        = 30
        match_thresh        = 0.8
        aspect_ratio_thresh = 1.6
        min_box_area        = 10
        cmc_method          = "sparseOptFlow"
        with_reid           = True
        fast_reid_config    = os.path.join(BOTSORT_ROOT, "fast_reid/configs/MOT17/sbs_S50.yml")
        fast_reid_weights   = os.path.join(BOTSORT_ROOT, "pretrained/mot17_sbs_S50.pth")
        proximity_thresh    = 0.5
        appearance_thresh   = 0.25
        device              = "cuda"
        fp16                = False
        name                = "botsort"
        ablation            = False
        mot20               = False

    for seq in SEQS:
        img_dir  = os.path.join(MOT17_ROOT, "train", seq, "img1")
        det_file = os.path.join(MOT17_ROOT, "train", seq, "det", "det.txt")
        seq_info = os.path.join(MOT17_ROOT, "train", seq, "seqinfo.ini")

        if not os.path.exists(img_dir):
            print(f"[WARN] img1 not found for {seq}, skipping.")
            continue

        if not os.path.exists(det_file):
            print(f"[WARN] det.txt not found for {seq}, skipping.")
            continue

        # Load pre-computed YOLOv8n detections
        det_dict = load_detections(det_file)
        print(f"  [det.txt] Loaded detections for {len(det_dict)} frames from {seq}")

        total_frames = 0
        if os.path.exists(seq_info):
            with open(seq_info) as f:
                for line in f:
                    if line.startswith("seqLength"):
                        total_frames = int(line.strip().split("=")[1])

        frames = sorted([f for f in os.listdir(img_dir)
                         if f.endswith('.jpg') or f.endswith('.png')])
        if total_frames == 0:
            total_frames = len(frames)

        print(f"\n{'='*60}")
        print(f"  Processing: {seq}  ({len(frames)} frames)")
        print(f"{'='*60}")

        result_file = os.path.join(track_results_dir, f"{seq}.txt")
        tracker = BoTSORT(Args(), frame_rate=30)

        with open(result_file, "w") as rf:
            for fi, frame_file in enumerate(frames):
                frame_id = int(frame_file.replace(".jpg","").replace(".png",""))
                img_path = os.path.join(img_dir, frame_file)

                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))

                # Use pre-computed YOLOv8n detections (no live YOLO)
                dets = det_dict.get(frame_id, [])
                dets_np = np.array(dets, dtype=float) if dets else np.empty((0, 6))

                online_targets = tracker.update(dets_np, frame)

                tracks = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid  = t.track_id
                    if tlwh[2] * tlwh[3] > 10:
                        x1, y1, w, h = tlwh
                        tracks.append([x1, y1, x1+w, y1+h, tid])
                        rf.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

                frame_resized = draw_tracks(frame_resized, tracks, seq, frame_id, total_frames)
                writer.write(frame_resized)

                if (fi+1) % 100 == 0:
                    print(f"  [{seq}] {fi+1}/{len(frames)} frames done...")

        print(f"  [{seq}] Done!")

    writer.release()
    print(f"\n  Video saved -> {video_path}")
    return track_results_dir

def setup_trackeval(track_results_dir):
    TRACKER_NAME = "BOTSORT_FASTREID"
    for seq in SEQS:
        dst_gt  = os.path.join(TRACKEVAL_DIR,"data","gt","mot_challenge","MOT17-train",seq,"gt")
        dst_trk = os.path.join(TRACKEVAL_DIR,"data","trackers","mot_challenge","MOT17-train",TRACKER_NAME,"data")
        os.makedirs(dst_gt,  exist_ok=True)
        os.makedirs(dst_trk, exist_ok=True)
        src_gt  = os.path.join(MOT17_ROOT,"train",seq,"gt","gt.txt")
        src_ini = os.path.join(MOT17_ROOT,"train",seq,"seqinfo.ini")
        src_trk = os.path.join(track_results_dir,f"{seq}.txt")
        if os.path.exists(src_gt):  shutil.copy(src_gt, dst_gt)
        if os.path.exists(src_ini): shutil.copy(src_ini, os.path.join(TRACKEVAL_DIR,"data","gt","mot_challenge","MOT17-train",seq))
        if os.path.exists(src_trk): shutil.copy(src_trk, dst_trk)

    seqmap_dir = os.path.join(TRACKEVAL_DIR,"data","gt","mot_challenge","seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    with open(os.path.join(seqmap_dir,"MOT17-train.txt"),"w") as f:
        f.write("name\n"+"\n".join(SEQS))
    return TRACKER_NAME

def run_evaluation(tracker_name):
    print("\n"+"="*60)
    print("  Running TrackEval...")
    print("="*60)
    script = os.path.join(TRACKEVAL_DIR,"scripts","run_mot_challenge.py")
    cmd = [
        sys.executable, script,
        "--BENCHMARK","MOT17",
        "--SPLIT_TO_EVAL","train",
        "--TRACKERS_TO_EVAL", tracker_name,
        "--METRICS","HOTA","CLEAR","Identity",
        "--GT_FOLDER", os.path.join(TRACKEVAL_DIR,"data","gt","mot_challenge"),
        "--TRACKERS_FOLDER", os.path.join(TRACKEVAL_DIR,"data","trackers","mot_challenge"),
        "--USE_PARALLEL","False",
        "--NUM_PARALLEL_CORES","1",
    ]
    subprocess.run(cmd, check=True)
    results_path = os.path.join(TRACKEVAL_DIR,"data","trackers","mot_challenge",
                                "MOT17-train",tracker_name,"pedestrian_summary.txt")
    if os.path.exists(results_path):
        print("\n"+"="*60)
        print("  FINAL SCORES")
        print("="*60)
        with open(results_path) as f:
            print(f.read())

if __name__ == "__main__":
    print("\n"+"="*60)
    print("  YOLOv8n det.txt + BoTSORT (FastReID SBS-S50) — MOT17")
    print("="*60)
    track_results_dir = run_tracking_and_video()
    tracker_name      = setup_trackeval(track_results_dir)
    run_evaluation(tracker_name)
    print("\n  ALL DONE!")