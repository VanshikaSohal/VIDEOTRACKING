# Tracking Results File Locations

Yeh document mein tino tracking approaches ke results kahan save hain, yeh likha hai.

---

## 1. Kalman Filter Based Approach (KALMAN-YOLO)

**Approach:** YOLOv8 Detection (pre-computed) + Kalman Filter Tracking

### Results Location:
```
KALMAN-YOLO/yolokfoutputs/track_results/
```

### Key Files:

| File | Full Path |
|------|-----------|
| **pedestrian_summary.txt** (Combined Results) | `KALMAN-YOLO/yolokfoutputs/track_results/pedestrian_summary.txt` |
| **pedestrian_detailed.csv** (Detailed Metrics) | `KALMAN-YOLO/yolokfoutputs/track_results/pedestrian_detailed.csv` |
| **Per-sequence Results** | `KALMAN-YOLO/yolokfoutputs/track_results/MOT17-XX-FRCNN.txt` |

### Combined Metrics (from pedestrian_summary.txt):
```
HOTA: 42.025
MOTA: 42.424
IDF1: 50.708
DetA: 38.601
AssA: 46.196
MOTP: 81.178
```

---

## 2. Diffusion Model Based Approach (DiffMOT)

**Approach:** YOLOv8 Detection (pre-computed) + Diffusion Model Motion Prediction (D²MP)

### Results Location:
```
DiffMOT-main/outputs/mot17/diffmot/
```

### Key Files:

| File | Full Path |
|------|-----------|
| **pedestrian_summary.txt** (Combined Results) | `DiffMOT-main/outputs/mot17/diffmot/pedestrian_summary.txt` |
| **pedestrian_detailed.csv** (Detailed Metrics) | `DiffMOT-main/outputs/mot17/diffmot/pedestrian_detailed.csv` |
| **Per-sequence Results** | `DiffMOT-main/outputs/mot17/diffmot/MOT17-XX-FRCNN.txt` |

### Combined Metrics (from pedestrian_summary.txt):
```
HOTA: 33.615
MOTA: 40.900
IDF1: 37.738
DetA: 39.435
AssA: 29.071
MOTP: 80.611
```

---

## 3. BoTSORT + FastReID Based Approach (BoTSORT-FastReID)

**Approach:** YOLOv8 Detection (pre-computed) + BoTSORT Tracker + FastReID SBS-S50 (ResNeSt50) ReID

### Results Location:
```
BOTSORT-FASTREID/TrackEval/data/trackers/mot_challenge/MOT17-train/BOTSORT_FASTREID/
```

### Key Files:

| File | Full Path |
|------|-----------|
| **pedestrian_summary.txt** (Combined Results) | `BOTSORT-FASTREID/TrackEval/data/trackers/mot_challenge/MOT17-train/BOTSORT_FASTREID/pedestrian_summary.txt` |
| **pedestrian_detailed.csv** (Detailed Metrics) | `BOTSORT-FASTREID/TrackEval/data/trackers/mot_challenge/MOT17-train/BOTSORT_FASTREID/pedestrian_detailed.csv` |
| **Per-sequence Results** | `BOTSORT-FASTREID/TrackEval/data/trackers/mot_challenge/MOT17-train/BOTSORT_FASTREID/MOT17-XX-FRCNN.txt` |

### Combined Metrics (from pedestrian_summary.txt):
```
HOTA: 43.609
MOTA: 40.828
IDF1: 52.656
DetA: 40.108
AssA: 47.920
MOTP: 80.531
```

---

## Quick Reference Table

| Approach | Framework | Results Path | pedestrian_summary.txt |
|----------|-----------|--------------|------------------------|
| **Kalman Filter** | KALMAN-YOLO | `KALMAN-YOLO/yolokfoutputs/track_results/` | HOTA: 42.025, MOTA: 42.424, IDF1: 50.708 |
| **Diffusion Model** | DiffMOT-main | `DiffMOT-main/outputs/mot17/diffmot/` | HOTA: 33.615, MOTA: 40.900, IDF1: 37.738 |
| **BoTSORT+FastReID** | BOTSORT-FASTREID | `BOTSORT-FASTREID/TrackEval/data/trackers/mot_challenge/MOT17-train/BOTSORT_FASTREID/` | HOTA: 43.609, MOTA: 40.828, IDF1: 52.656 |

---

## Summary

- **Best HOTA (Overall Tracking):** BoTSORT+FastReID (43.609)
- **Best MOTA (Detection):** KALMAN-YOLO (42.424)
- **Best IDF1 (Identity):** BoTSORT+FastReID (52.656)
- **Best MOTP (Localization):** KALMAN-YOLO (81.178)

---

## Detection Source Verification

All three trackers use **pre-computed YOLOv8n detections** (NOT live YOLO inference):

1. **KALMAN-YOLO:** Loads pre-computed detections from `MOT17/seq/det/det.txt` (see `yolomot.py` line 282)
2. **DiffMOT:** Loads pre-computed detections from `datasets/MOT17/seq/det_framewise/` (see `diffmot.py` line 107)
3. **BoTSORT+FastReID:** Loads pre-computed detections from `MOT17/seq/det/det.txt` (see `botsort_mot.py` line 33)
