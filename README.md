# VIDEOTRACKING

A comparative study of **classical**, **diffusion-based**, and **appearance-based** multi-object tracking on the MOT17 benchmark dataset. The repository contains three complete pipelines: YOLOv8 + Kalman Filter, YOLOv8 + Diffusion Model, and YOLOv8 + BoTSORT with FastReID SBS-S50 ReID backbone (ResNeSt50).

---

## Projects Overview

| | KALMAN-YOLO | DiffMOT-main | BoTSORT + FastReID |
|---|---|---|---|
| **Detection** | YOLOv8 | YOLOv8 | YOLOv8n |
| **Tracking** | Kalman Filter | Diffusion Model (D²MP) | BoTSORT with FastReID SBS-S50 ReID (ResNeSt50) |
| **Dataset** | MOT17 (7 sequences) | MOT17 (7 sequences) | MOT17 (7 sequences) |
| **Evaluation** | TrackEval (HOTA, CLEAR, Identity) | TrackEval (HOTA, CLEAR, Identity) | TrackEval (HOTA, CLEAR, Identity) |

---

## Repository Structure

```
VIDEOTRACKING/
├── BOTSORT-FASTREID/                  # BoTSORT + FastReID SBS-S50 (ResNeSt50) pipeline
│   ├── botsort_mot.py                  # BoTSORT tracking script with FastReID SBS-S50 (ResNeSt50) backbone
│   ├── eval_only.py                    # Optional evaluation helper
│   ├── BOTSORT-OUTPUT/
│   │   └── track_results/              # Per-sequence tracking .txt files
│   │       └── botsort_fastreid_tracking.mp4  # Tracking video output
│   └── TrackEval/                      # Evaluation toolkit (cloned)
│       └── data/trackers/mot_challenge/MOT17-train/BOTSORT_FASTREID/
│           ├── pedestrian_summary.txt  # Overall metric summary
│           ├── pedestrian_detailed.csv # Per-sequence metrics
│           └── (sequence .txt files)   # Per-sequence tracking results
│
├── KALMAN-YOLO/                        # YOLOv8 + Kalman Filter tracker
│   ├── yolomot.py                      # Main tracking script
│   ├── yoloeval.py                     # TrackEval evaluation wrapper
│   ├── requirements.txt                # Python dependencies
│   ├── mot17_yolo_trackeval_files.txt  # MOT17 sequence list for eval
│   ├── MOT17/                          # MOT17 dataset (7 sequences)
│   │   └── MOT17-XX-FRCNN/
│   │       ├── det/det.txt             # Per-frame detections (YOLOv8n pre-computed)
│   │       ├── gt/gt.txt               # Ground truth annotations
│   │       └── seqinfo.ini             # Sequence metadata
│   ├── TrackEval/                      # Evaluation toolkit (cloned)
│   │   └── data/trackers/mot_challenge/MOT17-train/YOLO_KF/
│   │       ├── pedestrian_summary.txt  # Overall metric summary
│   │       ├── pedestrian_detailed.csv # Per-sequence metrics
│   │       └── pedestrian_plot.png     # Metric plots
│   └── yolokfoutputs/
│       └── track_results/              # Per-sequence tracking .txt files
│
├── DiffMOT-main/                       # YOLOv8 + Diffusion Model tracker
│   ├── main.py                         # CLI entrypoint
│   ├── diffmot.py                      # Model trainer and evaluator
│   ├── requirement.txt                 # Python dependencies
│   ├── configs/                        # YAML experiment configs
│   │   ├── mot.yaml                    # MOT training config
│   │   └── mot17_test.yaml             # MOT17 test config
│   ├── models/                         # Model architecture
│   │   ├── diffusion.py                # Diffusion model utilities
│   │   ├── denoising_diffusion_pytorch.py  # Core diffusion model
│   │   ├── autoencoder.py              # Autoencoder module
│   │   └── condition_embedding.py      # Motion condition embedding
│   ├── tracker/                        # Tracker implementation
│   │   ├── DiffMOTtracker.py           # Main DiffMOT tracker class
│   │   └── matching.py                 # Association algorithms
│   ├── tracking_utils/                 # Helper utilities
│   │   └── kalman_filter.py            # Kalman filter utility (fallback)
│   ├── datasets/MOT17/                 # MOT17 dataset
│   ├── assets/diffmot_git.png          # Architecture diagram
│   └── outputs/mot17/diffmot/          # Evaluation outputs
│       ├── MOT17-XX-FRCNN.txt          # Per-sequence tracking results
│       ├── pedestrian_summary.txt      # Overall metric summary
│       ├── pedestrian_detailed.csv     # Per-sequence metrics
│       └── pedestrian_plot.png         # Metric plots
│
├── SWIN-TRANSFORMER/                  # Alternative BoTSORT output location (legacy)
│   └── BOTSORT-OUTPUT/
│       └── track_results/              # Per-sequence tracking .txt files
│
├── KALMAN-YOLO (TrackEval copy)/       # Additional TrackEval instance
├── README.md                           # This file (project overview)
├── RESULTS.md                          # Combined evaluation results
├── RESULTS_LOCATION.md                 # (DEPRECATED - use TRACKING_RESULTS_LOCATIONS.md)
└── TRACKING_RESULTS_LOCATIONS.md       # Detailed results file paths
```

---

## Architecture

### KALMAN-YOLO Pipeline

```
MOT17 Frames + YOLOv8n Pre-computed Detections (det.txt)
                                          ↓
                               Kalman Filter Prediction
                                          ↓
                               IoU-based Hungarian Matching
                                          ↓
                               Track ID Assignment → MOT17 Output .txt
```

**Detector:** Pre-computed YOLOv8n (not live inference)  
**Tracking:** Constant velocity Kalman Filter + Hungarian algorithm for data association

### DiffMOT Pipeline

![DiffMOT Architecture](DiffMOT-main/assets/diffmot_git.png)

```
MOT17 Frames + YOLOv8n Pre-computed Detections (framewise format)
                                          ↓
                          D²MP (Denoising Diffusion Motion Predictor)
                          Samples from N(0,I) + trajectory conditions
                                          ↓
                               Association Module
                                          ↓
                               Track ID Assignment → MOT17 Output .txt
```

**Detector:** Pre-computed YOLOv8n (framewise, not live inference)  
**Tracking:** Diffusion-based motion prediction + learned association

### BoTSORT + FastReID Pipeline

```
MOT17 Frames + YOLOv8n Pre-computed Detections (det.txt)
                                          ↓
                               FastReID SBS-S50 (ResNeSt50) ReID Embeddings
                                          ↓
                               BoTSORT Association: Hungarian Matching
                               (IoU + Cosine Similarity Fusion)
                                          ↓
                               Track ID Assignment → MOT17 Output .txt
```

**Detector:** Pre-computed YOLOv8n (not live inference)  
**ReID/Appearance:** FastReID with SBS-S50 backbone (ResNeSt50)  
**Motion:** Kalman Filter  
**Association:** Hungarian Algorithm with IoU + Cosine similarity fusion

---

## Results on MOT17

> Evaluated on 7 MOT17 training sequences: MOT17-02, 04, 05, 09, 10, 11, 13 (FRCNN detections)

### Combined Metrics Comparison

| Metric | KALMAN-YOLO | DiffMOT-main | BoTSORT + FastReID | Winner |
|--------|:-----------:|:------------:|:-------------------:|:------:|
| **HOTA** ↑ | 42.025 | 33.615 | **43.609** | BoTSORT + FastReID |
| **DetA** ↑ | 38.601 | 39.435 | **40.108** | BoTSORT + FastReID |
| **AssA** ↑ | 46.196 | 29.071 | **47.920** | BoTSORT + FastReID |
| **LocA** ↑ | **83.114** | 82.462 | 82.448 | KALMAN-YOLO |
| **MOTA** ↑ | **42.424** | 40.900 | 40.828 | KALMAN-YOLO |
| **MOTP** ↑ | **81.178** | 80.611 | 80.531 | KALMAN-YOLO |
| **IDF1** ↑ | 50.708 | 37.738 | **52.656** | BoTSORT + FastReID |

### Key Observations

- **BoTSORT + FastReID** achieves the highest HOTA (43.609), DetA (40.108), AssA (47.920), and IDF1 (52.656), demonstrating superior detection and association quality through FastReID SBS-S50 (ResNeSt50) ReID embeddings.
- **KALMAN-YOLO** maintains excellent localization (LocA: 83.114), detection accuracy (MOTA: 42.424), and pose tracking (MOTP: 81.178) as a strong classical baseline.
- **DiffMOT** shows balanced performance with respectable DetA (39.435) but lags in association (AssA: 29.071) and identity metrics (IDF1: 37.738), as diffusion-based motion prediction does not compensate for weaker ReID features.
- **FastReID SBS-S50 (ResNeSt50)**-powered BoTSORT significantly outperforms diffusion and Kalman-only approaches on association, identity, and overall tracking quality, confirming that stronger ReID embeddings are critical for MOT17 identity tracking.
- The results show that appearance-based association (BoTSORT + FastReID) excels at HOTA and IDF1, while classical Kalman filtering maintains superior localization accuracy.

---

## Output File Locations

### KALMAN-YOLO

| Output Type | Path |
|---|---|
| Tracking results (per sequence) | `KALMAN-YOLO/yolokfoutputs/track_results/*.txt` |
| Evaluation summary | `KALMAN-YOLO/yolokfoutputs/track_results/pedestrian_summary.txt` |
| Detailed metrics (CSV) | `KALMAN-YOLO/yolokfoutputs/track_results/pedestrian_detailed.csv` |
| Metric plots | `KALMAN-YOLO/yolokfoutputs/track_results/pedestrian_plot.png` |

### DiffMOT-main

| Output Type | Path |
|---|---|
| Tracking results (per sequence) | `DiffMOT-main/outputs/mot17/diffmot/*.txt` |
| Evaluation summary | `DiffMOT-main/outputs/mot17/diffmot/pedestrian_summary.txt` |
| Detailed metrics (CSV) | `DiffMOT-main/outputs/mot17/diffmot/pedestrian_detailed.csv` |
| Metric plots | `DiffMOT-main/outputs/mot17/diffmot/pedestrian_plot.png` |
| Terminal eval log | `DiffMOT-main/outputs/mot17/diffmot/TERMINALRESULTS.txt` |

### BoTSORT + FastReID

| Output Type | Path |
|---|---|
| Tracking results (per sequence) | `BOTSORT-FASTREID/BOTSORT-OUTPUT/track_results/*.txt` |
| Tracking video | `BOTSORT-FASTREID/BOTSORT-OUTPUT/botsort_fastreid_tracking.mp4` |
| Evaluation summary | `BOTSORT-FASTREID/TrackEval/data/trackers/mot_challenge/MOT17-train/BOTSORT_FASTREID/pedestrian_summary.txt` |
| Detailed metrics (CSV) | `BOTSORT-FASTREID/TrackEval/data/trackers/mot_challenge/MOT17-train/BOTSORT_FASTREID/pedestrian_detailed.csv` |

---

## Setup and Usage

### KALMAN-YOLO

```bash
cd KALMAN-YOLO
pip install -r requirements.txt

# Run tracker on MOT17
python yolomot.py

# Run evaluation
python yoloeval.py
```

> Requires: `yolov8x.pt` weights in `KALMAN-YOLO/` (not tracked in Git due to size)

### DiffMOT-main

```bash
cd DiffMOT-main
pip install -r requirement.txt

# Run tracking
python main.py --config configs/mot17_test.yaml
```

> Requires: Pretrained model weights in `DiffMOT-main/pretrained/` and MOT17 dataset in `DiffMOT-main/datasets/MOT17/`

### BoTSORT + FastReID (SBS-S50/ResNeSt50)

```bash
cd BOTSORT-FASTREID
pip install -r requirements.txt
pip install fastreid
pip install timm

# Run appearance-based BoTSORT tracking
python botsort_mot.py

# (Optional) Run evaluation with TrackEval
cd TrackEval
python scripts/run_mot_challenge.py
```

> Requires: BoTSORT + FastReID weights and MOT17 dataset available for tracking.  
> Output is written to `BOTSORT-FASTREID/BOTSORT-OUTPUT/track_results/` and evaluated results to `BOTSORT-FASTREID/TrackEval/data/trackers/mot_challenge/MOT17-train/BOTSORT_FASTREID/`

---

## Dataset

All pipelines use the **MOT17** benchmark dataset.

- 7 training sequences evaluated: MOT17-02, 04, 05, 09, 10, 11, 13
- Detector variant used: **FRCNN** (Faster R-CNN)
- Dataset not included in repository due to size — download from [MOTChallenge](https://motchallenge.net/data/MOT17/)

---

## Dependencies

### KALMAN-YOLO
- `ultralytics` — YOLOv8 detection
- `opencv-python` — video/frame processing
- `numpy`, `scipy` — numerical operations
- `TrackEval` — evaluation (cloned separately)

### DiffMOT-main
- `torch`, `torchvision` — deep learning framework
- `ultralytics` — YOLOv8
- `einops`, `tqdm` — model utilities
- `TrackEval` — evaluation

### BoTSORT + FastReID
- `torch`, `torchvision` — deep learning framework
- `ultralytics` — YOLOv8
- `fastreid` — ReID and SBS-S50 / ResNeSt50 backbone support
- `timm` — transformer backbone models
- `numpy` — numerical operations

---

## Project Structure and How to Run

**Project Overview:**
This project compares 3 multi-object tracking approaches on MOT17 benchmark (7 FRCNN sequences: MOT17-02, 04, 05, 09, 10, 11, 13).

**Three Trackers:**

1. KALMAN-YOLO (Classical): YOLOv8 detection + Kalman Filter tracking. Code in `KALMAN-YOLO/yolomot.py`.
2. DiffMOT (Diffusion-based): YOLOv8 detection + Diffusion Model motion prediction. Code in `DiffMOT-main/`.
3. BoTSORT + FastReID (Transformer-based): YOLOv8 detection + BoTSORT tracker with FastReID SBS-S50 (ResNeSt50) ReID backbone. Code in `botsort_mot.py`.

**Evaluation:**
All 3 trackers can be evaluated using TrackEval, with outputs stored in `KALMAN-YOLO/yolokfoutputs/track_results/`, `DiffMOT-main/outputs/mot17/diffmot/`, and `SWIN-TRANSFORMER/BOTSORT-OUTPUT/track_results/`.

---

## References

- [MOTChallenge Benchmark](https://motchallenge.net/)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DiffMOT Paper](https://arxiv.org/abs/2403.02075) — Lv et al., CVPR 2024
