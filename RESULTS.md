# Results Summary

This file collects available evaluation outputs (HOTA, MOTA, IDF1, etc.) for both projects and provides tables per-sequence and overall. Where numeric results are not present in the repository, entries are marked "N/A" and I provide the exact command to run TrackEval to generate them.

---

**KALMAN-YOLO**

- Detection outputs: not saved to disk by the main tracker script (detections are produced in-memory by YOLOv8 in `yolomot.py`).
- Final tracking outputs (MOTChallenge format): [KALMAN-YOLO/yolokfoutputs/track_results/](KALMAN-YOLO/yolokfoutputs/track_results/)
  - Files present: MOT17-02-FRCNN.txt, MOT17-04-FRCNN.txt, MOT17-05-FRCNN.txt, MOT17-09-FRCNN.txt, MOT17-10-FRCNN.txt, MOT17-11-FRCNN.txt, MOT17-13-FRCNN.txt
- Evaluation outputs (TrackEval): expected under `TrackEval` when `yoloeval.py` is run — TrackEval copies files into `TrackEval/data/trackers/mot_challenge/<TRACKER_NAME>/` and writes summary files (e.g. `pedestrian_summary.txt`) into `TrackEval/data/trackers/mot_challenge/<TRACKER_NAME>/`.
  - Example expected summary path: `TrackEval/data/trackers/mot_challenge/YOLO_KF/pedestrian_summary.txt`

KALMAN-YOLO — combined overall results (HOTA / MOTA / IDF1)

| Model | HOTA | MOTA | IDF1 | DetA | AssA | MOTP | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| KALMAN-YOLO (overall) | 42.025 | 42.424 | 50.708 | 38.601 | 46.196 | 81.178 | from `KALMAN-YOLO/yolokfoutputs/track_results/pedestrian_summary.txt` |

**Detector:** Pre-computed YOLOv8n (not live YOLO inference)  
**Tracking:** Kalman Filter with IoU-based Hungarian matching  
**Output:** MOTChallenge format (.txt files)

---

**DiffMOT-main**

- Detection outputs: configured input directories are under `DiffMOT-main/datasets/MOT17/` (the detector outputs are expected to be placed under `det_dir` / `datasets`); the training/eval config writes outputs to `outputs/mot17` (see `DiffMOT-main/configs/mot.yaml` `save_dir: outputs/mot17`).
- Final tracking outputs: available at `DiffMOT-main/outputs/mot17/diffmot/` in the repository.
- Evaluation outputs (TrackEval): the combined summary file is available at `DiffMOT-main/outputs/mot17/diffmot/pedestrian_summary.txt` and is used for the overall metric row below.

DiffMOT-main — combined overall results (HOTA / MOTA / IDF1)

| Model | HOTA | MOTA | IDF1 | DetA | AssA | MOTP | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| DiffMOT-main (overall) | 33.615 | 40.900 | 37.738 | 39.435 | 29.071 | 80.611 | from `DiffMOT-main/outputs/mot17/diffmot/pedestrian_summary.txt` |

**Detector:** Pre-computed YOLOv8n detections (framewise format)  
**Tracking:** Diffusion-based Motion Prediction (D²MP) with association  
**Output:** MOTChallenge format (.txt files)

---

**BOTSORT + FastReID**

- Evaluation outputs: summary file available at `SWIN-TRANSFORMER/BOTSORT-OUTPUT/track_results/pedestrian_summary.txt`.
- Only the combined overall metrics are available in the repo file.

BOTSORT + FastReID — overall results (HOTA / MOTA / IDF1)

| Sequence | HOTA | MOTA | IDF1 | DetA | AssA | MOTP | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| **Overall (train)** | 43.609 | 40.828 | 52.656 | 40.108 | 47.920 | 80.531 | from `BOTSORT-FASTREID/TrackEval/data/trackers/mot_challenge/MOT17-train/BOTSORT_FASTREID/pedestrian_summary.txt` |

**Detector:** Pre-computed YOLOv8n (not live YOLO inference)  
**Tracking:** BoTSORT with FastReID SBS-S50 (ResNeSt50) ReID backbone  
**Output:** MOTChallenge format (.txt files)

---

Summary of findings

- I found concrete final tracking output files for `KALMAN-YOLO` in `KALMAN-YOLO/yolokfoutputs/track_results/` (these are raw tracker outputs in MOTChallenge format), and I used `KALMAN-YOLO/yolokfoutputs/track_results/pedestrian_summary.txt` to populate the KALMAN-YOLO table.
- I found combined evaluation summaries for the other approaches in the repository: `DiffMOT-main/outputs/mot17/diffmot/pedestrian_summary.txt` and `SWIN-TRANSFORMER/BOTSORT-OUTPUT/track_results/pedestrian_summary.txt`.

Next steps I can take (pick one):

- Run the KALMAN-YOLO evaluation locally (execute `python yoloeval.py`) and populate `RESULTS.md` with the numbers.
- Run the DiffMOT evaluation locally (execute the TrackEval script) after `outputs/mot17` is populated, and then populate `RESULTS.md`.

Tell me which evaluation to run now (I can run `yoloeval.py` for KALMAN-YOLO first), or provide existing summary files to extract and I'll parse them into the tables.