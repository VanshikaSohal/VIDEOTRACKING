import os, sys, shutil, glob, subprocess

TRACKEVAL_DIR = "TrackEval"
MOT17_ROOT    = "MOT17"
OUTPUT_ROOT   = "yolokfoutputs"
TRACKER_NAME  = "YOLO_KF"

SEQS = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]

track_results_dir = os.path.join(OUTPUT_ROOT, "track_results")

# Copy all GT + tracking files
for seq in SEQS:
    dst_gt  = os.path.join(TRACKEVAL_DIR,"data","gt","mot_challenge","MOT17-train",seq,"gt")
    dst_ini = os.path.join(TRACKEVAL_DIR,"data","gt","mot_challenge","MOT17-train",seq)
    dst_trk = os.path.join(TRACKEVAL_DIR,"data","trackers","mot_challenge","MOT17-train",TRACKER_NAME,"data")
    os.makedirs(dst_gt, exist_ok=True)
    os.makedirs(dst_trk, exist_ok=True)
    shutil.copy(os.path.join(MOT17_ROOT, seq, "gt", "gt.txt"), dst_gt)
    shutil.copy(os.path.join(MOT17_ROOT, seq, "seqinfo.ini"), dst_ini)
    shutil.copy(os.path.join(track_results_dir, f"{seq}.txt"), dst_trk)

# Seqmap — all 7
seqmap_dir = os.path.join(TRACKEVAL_DIR,"data","gt","mot_challenge","seqmaps")
os.makedirs(seqmap_dir, exist_ok=True)
with open(os.path.join(seqmap_dir,"MOT17-train.txt"),"w") as f:
    f.write("name\n" + "\n".join(SEQS))

# Numpy fix
for path in glob.glob(os.path.join(TRACKEVAL_DIR,"**","*.py"), recursive=True):
    with open(path,"r",errors="ignore") as f:
        code = f.read()
    code = code.replace("np.float,","float,").replace("np.float)","float)") \
               .replace("np.int,","int,").replace("np.int)","int)") \
               .replace("np.bool,","bool,").replace("np.bool)","bool)")
    with open(path,"w") as f:
        f.write(code)

# Run eval
script = r"C:\Users\vansh\VIDEOTRACKING\KALMAN-YOLO\TrackEval\scripts\run_mot_challenge.py"
subprocess.run([sys.executable, script,
    "--BENCHMARK","MOT17",
    "--SPLIT_TO_EVAL","train",
    "--TRACKERS_TO_EVAL", TRACKER_NAME,
    "--METRICS","HOTA","CLEAR","Identity",
    "--GT_FOLDER", os.path.join(TRACKEVAL_DIR,"data","gt","mot_challenge"),
    "--TRACKERS_FOLDER", os.path.join(TRACKEVAL_DIR,"data","trackers","mot_challenge"),
    "--USE_PARALLEL","False",
], check=True)