import sys
import os
from multiprocessing import freeze_support
import codecs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval
if __name__ == '__main__':
    freeze_support()
    # -------------------------
    # EVALUATOR
    # -------------------------
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    evaluator = trackeval.Evaluator(eval_config)
    # -------------------------
    # DATASET CONFIG
    # -------------------------
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ROOT = r"C:\Users\vansh\VANSHIKASOHAL\DiffMOT-main"
    ROOT = r"C:\Users\vansh\VANSHIKASOHAL\DiffMOT-main"
    KF_ROOT = r"C:\Users\vansh\VIDEOTRACKING\KALMAN-YOLO"
    dataset_config['GT_FOLDER'] = os.path.join(ROOT, "datasets", "MOT17", "train")
    dataset_config['TRACKERS_FOLDER'] = os.path.join(KF_ROOT, "yolokfoutputs")
    dataset_config['OUTPUT_FOLDER'] = os.path.join(KF_ROOT, "yolokfoutputs")
    dataset_config['TRACKER_SUB_FOLDER'] = ""
    dataset_config['BENCHMARK'] = "MOT17"
    dataset_config['SPLIT_TO_EVAL'] = "train"
    dataset_config['SKIP_SPLIT_FOL'] = True
    dataset_config['TRACKERS_TO_EVAL'] = ['track_results']
    dataset_config['SEQMAP_FOLDER'] = os.path.join(ROOT, "datasets", "MOT17", "seqmaps")
    dataset_config['SEQMAP_FILE'] = None
    dataset_config['SEQ_INFO'] = {
    "MOT17-02-FRCNN": None,
    "MOT17-04-FRCNN": None,
    "MOT17-05-FRCNN": None,
    "MOT17-09-FRCNN": None,
    "MOT17-10-FRCNN": None,
    "MOT17-11-FRCNN": None,
    "MOT17-13-FRCNN": None,
    }
    # -------------------------
    # METRICS
    # -------------------------
    metrics_config = {
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],
        'THRESHOLD': 0.5
    }
    metrics_list = []
    for metric in [
        trackeval.metrics.HOTA,
        trackeval.metrics.CLEAR,
        trackeval.metrics.Identity
    ]:
        metrics_list.append(metric(metrics_config))
    # -------------------------
    # CLEAN SEQMAP READ (IMPORTANT FIX)
    # -------------------------
    seqmap_path = os.path.join(ROOT, "datasets", "MOT17", "seqmaps", "MOT17-train.txt")
    if not os.path.isfile(seqmap_path):
        raise FileNotFoundError("SEQMAP NOT FOUND: " + seqmap_path)
    with codecs.open(seqmap_path, "r", "utf-8-sig") as f:
        seq_list = [x.strip() for x in f.readlines() if x.strip()]
    print("SEQ LIST LOADED:", seq_list)
    # -------------------------
    # RUN
    # -------------------------
    print("SEQMAP FULL PATH:", os.path.join(ROOT, "datasets", "MOT17", "seqmaps", "MOT17-train.txt"))
    print("DATASET CONFIG:", dataset_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    print(ROOT)
    print(os.path.exists(seqmap_path))
    evaluator.evaluate(dataset_list, metrics_list)