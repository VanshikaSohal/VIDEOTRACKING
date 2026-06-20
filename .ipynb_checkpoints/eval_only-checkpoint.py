import sys
sys.path.insert(0, r'C:\Users\vansh\VIDEOTRACKING\DiffMOT-main\TrackEval')
import os
import shutil

TRACKEVAL_DIR = r"C:\Users\vansh\VIDEOTRACKING\DiffMOT-main\TrackEval"
MOT17_ROOT    = r"C:\Users\vansh\VANSHIKASOHAL\DiffMOT-main\datasets\MOT17"
track_results_dir = r"C:\Users\vansh\VIDEOTRACKING\BOTSORT-OUTPUT\track_results"
TRACKER_NAME  = "BOTSORT_SWIN"

SEQS = [
    "MOT17-02-FRCNN","MOT17-04-FRCNN","MOT17-05-FRCNN",
    "MOT17-09-FRCNN","MOT17-10-FRCNN","MOT17-11-FRCNN","MOT17-13-FRCNN",
]

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
    else: print(f"MISSING: {src_trk}")

seqmap_dir = os.path.join(TRACKEVAL_DIR,"data","gt","mot_challenge","seqmaps")
os.makedirs(seqmap_dir, exist_ok=True)
with open(os.path.join(seqmap_dir,"MOT17-train.txt"),"w") as f:
    f.write("name\n"+"\n".join(SEQS))

print("Files copied! Now running eval...")

import trackeval
evaluator = trackeval.Evaluator({
    'USE_PARALLEL': False,
    'NUM_PARALLEL_CORES': 1,
    'BREAK_ON_ERROR': True,
    'PRINT_RESULTS': True,
    'PRINT_CONFIG': True,
    'TIME_PROGRESS': True,
    'OUTPUT_SUMMARY': True,
    'OUTPUT_DETAILED': True,
    'PLOT_CURVES': False,
})
dataset_config = {
    'GT_FOLDER': os.path.join(TRACKEVAL_DIR,"data","gt","mot_challenge"),
    'TRACKERS_FOLDER': os.path.join(TRACKEVAL_DIR,"data","trackers","mot_challenge"),
    'BENCHMARK': 'MOT17',
    'SPLIT_TO_EVAL': 'train',
    'TRACKERS_TO_EVAL': [TRACKER_NAME],
    'CLASSES_TO_EVAL': ['pedestrian'],
    'DO_PREPROC': False,
}
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
evaluator.evaluate(dataset_list, metrics_list)