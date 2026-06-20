# Combined Tracking Results (Kalman, DiffMOT, BotSort)

Date: 2026-06-06

This file collects the available outputs for the three tracking approaches in this workspace and points to where numeric summaries (TrackEval outputs) should be written.

## Summary Status
- **KALMAN (YOLO + KF):** Outputs present — [KALMAN-YOLO/yolokfoutputs/track_results/](KALMAN-YOLO/yolokfoutputs/track_results/)
- **Diffusion (DiffMOT):** Outputs expected at `DiffMOT-main/outputs/mot17/` — not present in repo (marked N/A).
- **BOTSORT:** Outputs present — [BOTSORT-OUTPUT/track_results/](BOTSORT-OUTPUT/track_results/)

## Per-approach details

### KALMAN-YOLO
- Raw tracker outputs: [KALMAN-YOLO/yolokfoutputs/track_results/](KALMAN-YOLO/yolokfoutputs/track_results/)
- Recommended summary path (TrackEval): `TrackEval/data/trackers/mot_challenge/YOLO_KF/pedestrian_summary.txt`

### DiffMOT-main
- Expected outputs directory: `DiffMOT-main/outputs/mot17/`
- Current status: no `outputs/mot17/` directory found in repository. Run DiffMOT training/eval to produce tracker outputs.

### BOTSORT
- Tracker outputs: [BOTSORT-OUTPUT/track_results/](BOTSORT-OUTPUT/track_results/)
- These files are already in MOTChallenge format (MOT17-*.txt).

## Summary table (fill after running TrackEval)

| Approach | HOTA | MOTA | IDF1 | Notes |
|---|---:|---:|---:|---|
| YOLO + KF | N/A | N/A | N/A | See KALMAN-YOLO outputs link above |
| DiffMOT (diffusion) | N/A | N/A | N/A | Outputs missing; run DiffMOT to generate |
| BotSort | N/A | N/A | N/A | See BOTSORT-OUTPUT/track_results/ |

## How to generate numeric summaries (quick)

1. Produce tracker outputs (MOTChallenge format) for each method and place them under an outputs folder, e.g.:

```bash
# Example (run from project root)
# For DiffMOT: ensure `DiffMOT-main/configs/mot.yaml` points to dataset and run the eval/train script to write outputs to `DiffMOT-main/outputs/mot17/`.
# For KALMAN-YOLO and BotSort the outputs are already under their track_results folders.
```

2. Run TrackEval (or the project's provided TrackEval wrapper) to compute HOTA/MOTA/IDF1 and write a summary text file into each tracker folder.

3. Copy the numeric results into the table above.

---

If you want, I can (A) auto-populate the table by running TrackEval here for the available tracker outputs, or (B) create a small script to collect per-tracker summary files into this combined markdown file. Which do you prefer?
