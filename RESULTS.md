# Results Summary

This file collects available evaluation outputs (HOTA, MOTA, IDF1, etc.) for both projects and provides tables per-sequence and overall. Where numeric results are not present in the repository, entries are marked "N/A" and I provide the exact command to run TrackEval to generate them.

---

**KALMAN-YOLO**

- Detection outputs: not saved to disk by the main tracker script (detections are produced in-memory by YOLOv8 in `yolomot.py`).
- Final tracking outputs (MOTChallenge format): [KALMAN-YOLO/yolokfoutputs/track_results/](KALMAN-YOLO/yolokfoutputs/track_results/)
  - Files present: MOT17-02-FRCNN.txt, MOT17-04-FRCNN.txt, MOT17-05-FRCNN.txt, MOT17-09-FRCNN.txt, MOT17-10-FRCNN.txt, MOT17-11-FRCNN.txt, MOT17-13-FRCNN.txt
- Evaluation outputs (TrackEval): expected under `TrackEval` when `yoloeval.py` is run — TrackEval copies files into `TrackEval/data/trackers/mot_challenge/<TRACKER_NAME>/` and writes summary files (e.g. `pedestrian_summary.txt`) into `TrackEval/data/trackers/mot_challenge/<TRACKER_NAME>/`.
  - Example expected summary path: `TrackEval/data/trackers/mot_challenge/YOLO_KF/pedestrian_summary.txt`

KALMAN-YOLO — per-sequence results (HOTA / MOTA / IDF1)

| Sequence | HOTA | MOTA | IDF1 | Notes |
|---|---:|---:|---:|---|
| MOT17-02-FRCNN | N/A | N/A | N/A | no summary file found in repo |
| MOT17-04-FRCNN | N/A | N/A | N/A | no summary file found in repo |
| MOT17-05-FRCNN | N/A | N/A | N/A | no summary file found in repo |
| MOT17-09-FRCNN | N/A | N/A | N/A | no summary file found in repo |
| MOT17-10-FRCNN | N/A | N/A | N/A | no summary file found in repo |
| MOT17-11-FRCNN | N/A | N/A | N/A | no summary file found in repo |
| MOT17-13-FRCNN | N/A | N/A | N/A | no summary file found in repo |
| **Overall (train)** | N/A | N/A | N/A | no combined summary found |

Notes: To compute and save these results, run (from `KALMAN-YOLO`):

```bash
python yoloeval.py
```

This will copy tracker files into `TrackEval/data/...` and invoke `TrackEval/scripts/run_mot_challenge.py`, which will produce `pedestrian_summary.txt` and `pedestrian_detailed.csv` under the TrackEval tracker folder.

---

**DiffMOT-main**

- Detection outputs: configured input directories are under `DiffMOT-main/datasets/MOT17/` (the detector outputs are expected to be placed under `det_dir` / `datasets`); the training/eval config writes outputs to `outputs/mot17` (see `DiffMOT-main/configs/mot.yaml` `save_dir: outputs/mot17`).
- Final tracking outputs: expected at `DiffMOT-main/outputs/mot17/` (per `configs/mot.yaml` and TrackEval script). There is currently no `outputs/` directory in the repo state.
- Evaluation outputs (TrackEval): `DiffMOT-main/TrackEval` is included; when `run_mot_challenge.py` (in `DiffMOT-main/TrackEval/scripts/`) is executed it writes summary files into the tracker output folder (configured as `outputs/mot17/<tracker>/`).

DiffMOT-main — per-sequence results (HOTA / MOTA / IDF1)

| Sequence | HOTA | MOTA | IDF1 | Notes |
|---|---:|---:|---:|---|
| MOT17-02-FRCNN | N/A | N/A | N/A | outputs/mot17 not found in repo |
| MOT17-04-FRCNN | N/A | N/A | N/A | outputs/mot17 not found in repo |
| MOT17-05-FRCNN | N/A | N/A | N/A | outputs/mot17 not found in repo |
| MOT17-09-FRCNN | N/A | N/A | N/A | outputs/mot17 not found in repo |
| MOT17-10-FRCNN | N/A | N/A | N/A | outputs/mot17 not found in repo |
| MOT17-11-FRCNN | N/A | N/A | N/A | outputs/mot17 not found in repo |
| MOT17-13-FRCNN | N/A | N/A | N/A | outputs/mot17 not found in repo |
| **Overall (train)** | N/A | N/A | N/A | no combined summary found |

Notes: To run DiffMOT evaluation (from `DiffMOT-main/TrackEval`):

```bash
# from repository root
cd DiffMOT-main/TrackEval
python scripts/run_mot_challenge.py
```

This script expects `dataset_config` in `run_mot_challenge.py` to point to the correct `ROOT` and will generate `pedestrian_summary.txt` and `pedestrian_detailed.csv` under the configured `outputs/mot17/<tracker>/` folder.

---

Summary of findings

- I found concrete final tracking output files for `KALMAN-YOLO` in `KALMAN-YOLO/yolokfoutputs/track_results/` (these are raw tracker outputs in MOTChallenge format).
- I did not find any computed TrackEval summary files (`*_summary.txt` or `*_detailed.csv`) for either project in the repository. Therefore all metric table cells are currently marked `N/A`.

Next steps I can take (pick one):

- Run the KALMAN-YOLO evaluation locally (execute `python yoloeval.py`) and populate `RESULTS.md` with the numbers.
- Run the DiffMOT evaluation locally (execute the TrackEval script) after `outputs/mot17` is populated, and then populate `RESULTS.md`.

Tell me which evaluation to run now (I can run `yoloeval.py` for KALMAN-YOLO first), or provide existing summary files to extract and I'll parse them into the tables.