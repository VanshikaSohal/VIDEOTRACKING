# VIDEOTRACKING — Complete Project Structure

This repository implements and evaluates three multi-object tracking pipelines on MOT17,
all using YOLOv8 for detection. Results are compared in `RESULTS.md`.

---

## Top-level

```
VIDEOTRACKING/
├── .gitignore
├── PROJECT_STRUCTURE.md
├── README.md
├── RESULTS.md                        ← cross-pipeline metrics comparison
├── TRACKING_RESULTS_LOCATIONS.md     ← guide to every output file location
├── KALMAN-YOLO/                      ← Pipeline 1: YOLOv8 + Kalman Filter (baseline)
├── DiffMOT-main/                     ← Pipeline 2: YOLOv8 + Diffusion tracker (D²MP)
├── BOTSORT-FASTREID/                 ← Pipeline 3: YOLOv8 + BoTSORT + FastReID SBS-S50
└── SWIN-TRANSFORMER/                 ← Swin Transformer experiment outputs only
```

---

## `KALMAN-YOLO/`

Classical baseline: YOLOv8 detections fed into a Kalman Filter tracker.

```
KALMAN-YOLO/
├── yolomot.py                        ← main tracking script
├── yoloeval.py                       ← TrackEval evaluation wrapper
├── requirements.txt
│
├── MOT17/                            ← MOT17 dataset (git-ignored)
│
├── TrackEval/                        ← evaluation library
│   ├── .gitignore
│   ├── LICENSE
│   ├── Readme.md
│   ├── minimum_requirements.txt
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── setup.cfg
│   ├── setup.py
│   ├── data/
│   │   ├── gt/
│   │   │   └── mot_challenge/
│   │   │       ├── seqmaps/
│   │   │       │   └── MOT17-train.txt
│   │   │       └── MOT17-train/
│   │   │           ├── MOT17-02-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           ├── MOT17-04-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           ├── MOT17-05-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           ├── MOT17-09-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           ├── MOT17-10-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           ├── MOT17-11-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           └── MOT17-13-FRCNN/  (gt/ + seqinfo.ini)
│   │   └── trackers/
│   │       └── mot_challenge/
│   │           └── MOT17-train/
│   │               └── YOLO_KF/
│   │                   └── data/
│   │                       ├── MOT17-02-FRCNN.txt
│   │                       ├── MOT17-04-FRCNN.txt
│   │                       ├── MOT17-05-FRCNN.txt
│   │                       ├── MOT17-09-FRCNN.txt
│   │                       ├── MOT17-10-FRCNN.txt
│   │                       ├── MOT17-11-FRCNN.txt
│   │                       └── MOT17-13-FRCNN.txt
│   ├── docs/
│   │   ├── BDD100k-format.txt
│   │   ├── DAVIS-format.txt
│   │   ├── KITTI-format.txt
│   │   ├── MOTChallenge-format.txt
│   │   ├── MOTS-format.txt
│   │   ├── TAO-format.txt
│   │   ├── YouTube-VIS-format.txt
│   │   ├── How_To/
│   │   │   └── Add_a_new_metric.md
│   │   ├── MOTChallenge-Official/
│   │   │   └── Readme.md
│   │   ├── OpenWorldTracking-Official/
│   │   │   └── Readme.md
│   │   └── RobMOTS-Official/
│   │       └── Readme.md
│   ├── scripts/
│   │   ├── comparison_plots.py
│   │   ├── python
│   │   ├── run_bdd.py
│   │   ├── run_burst.py
│   │   ├── run_burst_ow.py
│   │   ├── run_davis.py
│   │   ├── run_headtracking_challenge.py
│   │   ├── run_kitti.py
│   │   ├── run_kitti_mots.py
│   │   ├── run_mot_challenge.py
│   │   ├── run_mots_challenge.py
│   │   ├── run_person_path_22.py
│   │   ├── run_rob_mots.py
│   │   ├── run_tao.py
│   │   ├── run_tao_ow.py
│   │   └── run_youtube_vis.py
│   ├── tests/
│   │   ├── test_all_quick.py
│   │   ├── test_davis.py
│   │   ├── test_metrics.py
│   │   ├── test_mot17.py
│   │   └── test_mots.py
│   └── trackeval/
│       ├── __init__.py
│       ├── _timing.py
│       ├── eval.py
│       ├── plotting.py
│       ├── utils.py
│       ├── baselines/
│       │   ├── __init__.py
│       │   ├── baseline_utils.py
│       │   ├── non_overlap.py
│       │   ├── pascal_colormap.py
│       │   ├── stp.py
│       │   ├── thresholder.py
│       │   └── vizualize.py
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── _base_dataset.py
│       │   ├── bdd100k.py
│       │   ├── burst.py
│       │   ├── burst_ow.py
│       │   ├── davis.py
│       │   ├── head_tracking_challenge.py
│       │   ├── kitti_2d_box.py
│       │   ├── kitti_mots.py
│       │   ├── mot_challenge_2d_box.py
│       │   ├── mots_challenge.py
│       │   ├── person_path_22.py
│       │   ├── rob_mots.py
│       │   ├── rob_mots_classmap.py
│       │   ├── run_rob_mots.py
│       │   ├── tao.py
│       │   ├── tao_ow.py
│       │   ├── youtube_vis.py
│       │   └── burst_helpers/
│       │       ├── __init__.py
│       │       ├── burst_base.py
│       │       ├── burst_ow_base.py
│       │       ├── BURST_SPECIFIC_ISSUES.md
│       │       ├── convert_burst_format_to_tao_format.py
│       │       ├── format_converter.py
│       │       └── tao_categories.json
│       └── metrics/
│           ├── __init__.py
│           ├── _base_metric.py
│           ├── clear.py
│           ├── count.py
│           ├── hota.py
│           ├── identity.py
│           ├── ideucl.py
│           ├── j_and_f.py
│           ├── track_map.py
│           └── vace.py
│
└── yolokfoutputs/
    ├── TRACKING.txt
    ├── yolo_kf_tracking_all_sequences.mp4
    └── track_results/
        ├── MOT17-02-FRCNN.txt
        ├── MOT17-04-FRCNN.txt
        ├── MOT17-05-FRCNN.txt
        ├── MOT17-09-FRCNN.txt
        ├── MOT17-10-FRCNN.txt
        ├── MOT17-11-FRCNN.txt
        ├── MOT17-13-FRCNN.txt
        ├── pedestrian_summary.txt
        ├── pedestrian_detailed.csv
        ├── pedestrian_plot.pdf
        └── pedestrian_plot.png
```

---

## `DiffMOT-main/`

Diffusion model-based tracker (D²MP). Uses a learned diffusion motion model instead of
a Kalman Filter for motion prediction.

```
DiffMOT-main/
├── main.py                           ← entry point for training / testing
├── diffmot.py                        ← core DiffMOT model and tracking logic
├── mot_data_process.py               ← MOT dataset preprocessing
├── dancetrack_data_process.py        ← DanceTrack preprocessing
├── sports_data_process.py            ← SportsMOT preprocessing
├── createdetfor4.py                  ← detection data format converter
├── convert_mot17_to_framewise.py     ← converts MOT17 to framewise format
├── requirement.txt
├── TRACKING.txt
├── LICENSE
├── README.md
├── CODEFILE.ipynb
│
├── configs/
│   ├── mot.yaml
│   ├── mot17_test.yaml
│   ├── mot20_test.yaml
│   ├── dancetrack.yaml
│   ├── dancetrack_test.yaml
│   ├── sportsmot.yaml
│   └── sportsmot_test.yaml
│
├── models/
│   ├── autoencoder.py
│   ├── common.py
│   ├── condition_embedding.py
│   ├── denoising_diffusion_pytorch.py
│   └── diffusion.py
│
├── tracker/
│   ├── DiffMOTtracker.py
│   ├── matching.py
│   ├── basetrack.py
│   ├── cmc.py
│   ├── gmc.py
│   ├── embedding.py
│   └── ocsort_tracker/
│       ├── association.py
│       ├── kalmanfilter.py
│       └── ocsort.py
│
├── tracking_utils/
│   ├── evaluation.py
│   ├── io.py
│   ├── kalman_filter.py
│   ├── NSA_kalman_filter.py
│   ├── log.py
│   ├── nms.py
│   ├── parse_config.py
│   ├── timer.py
│   ├── utils.py
│   └── visualization.py
│
├── external/
│   ├── adaptors/
│   │   ├── __init__.py
│   │   └── fastreid_adaptor.py
│   └── fastreid/                     ← included FastReID library
│       ├── __init__.py
│       ├── .gitignore
│       ├── CHANGELOG.md
│       ├── GETTING_STARTED.md
│       ├── INSTALL.md
│       ├── LICENSE
│       ├── MODEL_ZOO.md
│       ├── README.md
│       ├── .github/
│       │   ├── FastReID-Logo.png
│       │   ├── wechat_group.png
│       │   ├── ISSUE_TEMPLATE/
│       │   │   ├── bugs.md
│       │   │   ├── config.yml
│       │   │   ├── questions-help-support.md
│       │   │   └── unexpected-problems-bugs.md
│       │   └── workflows/
│       │       ├── issue_auto_close.yml
│       │       └── lint_python.yml
│       ├── configs/
│       │   ├── Base-AGW.yml
│       │   ├── Base-bagtricks.yml
│       │   ├── Base-MGN.yml
│       │   ├── Base-SBS.yml
│       │   ├── DukeMTMC/  (AGW_R50.yml, AGW_S50.yml, sbs_S50.yml, ...)
│       │   ├── Market1501/  (AGW_R50.yml, sbs_S50.yml, bagtricks_vit.yml, ...)
│       │   ├── MSMT17/  (AGW_R50.yml, sbs_S50.yml, ...)
│       │   ├── VehicleID/  (bagtricks_R50-ibn.yml)
│       │   ├── VeRi/  (sbs_R50-ibn.yml)
│       │   └── VERIWild/  (bagtricks_R50-ibn.yml)
│       ├── datasets/
│       │   └── README.md
│       ├── demo/
│       │   ├── demo.py
│       │   ├── predictor.py
│       │   ├── visualize_result.py
│       │   ├── plot_roc_with_pickle.py
│       │   └── README.md
│       ├── docker/
│       │   ├── Dockerfile
│       │   └── README.md
│       ├── docs/
│       │   ├── .gitignore
│       │   ├── conf.py
│       │   ├── index.rst
│       │   ├── Makefile
│       │   ├── README.md
│       │   ├── requirements.txt
│       │   ├── modules/  (checkpoint.rst, config.rst, data.rst, engine.rst, ...)
│       │   └── _static/css/
│       ├── fastreid/                 ← FastReID core package
│       │   ├── __init__.py
│       │   ├── config/
│       │   │   ├── __init__.py
│       │   │   ├── config.py
│       │   │   └── defaults.py
│       │   ├── data/
│       │   │   ├── __init__.py
│       │   │   ├── build.py
│       │   │   ├── common.py
│       │   │   ├── data_utils.py
│       │   │   ├── datasets/
│       │   │   ├── samplers/
│       │   │   └── transforms/
│       │   ├── engine/
│       │   │   ├── __init__.py
│       │   │   ├── defaults.py
│       │   │   ├── hooks.py
│       │   │   ├── launch.py
│       │   │   └── train_loop.py
│       │   ├── evaluation/
│       │   │   ├── __init__.py
│       │   │   ├── clas_evaluator.py
│       │   │   ├── evaluator.py
│       │   │   ├── query_expansion.py
│       │   │   ├── rank.py
│       │   │   ├── reid_evaluation.py
│       │   │   ├── rerank.py
│       │   │   ├── roc.py
│       │   │   ├── testing.py
│       │   │   └── rank_cylib/
│       │   ├── layers/
│       │   │   ├── __init__.py
│       │   │   ├── activation.py
│       │   │   ├── any_softmax.py
│       │   │   ├── batch_norm.py
│       │   │   ├── context_block.py
│       │   │   ├── drop.py
│       │   │   ├── frn.py
│       │   │   ├── gather_layer.py
│       │   │   ├── helpers.py
│       │   │   ├── non_local.py
│       │   │   ├── pooling.py
│       │   │   ├── se_layer.py
│       │   │   ├── splat.py
│       │   │   └── weight_init.py
│       │   ├── modeling/
│       │   │   ├── __init__.py
│       │   │   ├── backbones/
│       │   │   ├── heads/
│       │   │   ├── losses/
│       │   │   └── meta_arch/
│       │   ├── solver/
│       │   │   ├── __init__.py
│       │   │   ├── build.py
│       │   │   ├── lr_scheduler.py
│       │   │   └── optim/
│       │   └── utils/
│       │       ├── __init__.py
│       │       ├── checkpoint.py
│       │       ├── collect_env.py
│       │       ├── comm.py
│       │       ├── compute_dist.py
│       │       ├── env.py
│       │       ├── events.py
│       │       ├── faiss_utils.py
│       │       ├── file_io.py
│       │       ├── history_buffer.py
│       │       ├── logger.py
│       │       ├── params.py
│       │       ├── precision_bn.py
│       │       ├── registry.py
│       │       ├── summary.py
│       │       ├── timer.py
│       │       └── visualizer.py
│       ├── projects/
│       │   ├── README.md
│       │   ├── CrossDomainReID/  (README.md)
│       │   ├── DG-ReID/  (README.md)
│       │   ├── FastAttr/  (configs/, fastattr/, README.md, train_net.py)
│       │   ├── FastClas/  (configs/, fastclas/, README.md, train_net.py)
│       │   ├── FastDistill/  (configs/, fastdistill/, README.md, train_net.py)
│       │   ├── FastFace/  (configs/, fastface/, README.md, train_net.py)
│       │   ├── FastRetri/  (configs/, fastretri/, README.md, train_net.py)
│       │   ├── FastRT/  (demo/, docker/, fastrt/, include/, pybind_interface/, tools/, CMakeLists.txt, README.md)
│       │   ├── FastTune/  (autotuner/, configs/, README.md, tune_net.py)
│       │   ├── HAA/  (Readme.md)
│       │   ├── NAIC20/  (configs/, naic/, label.txt, README.md, train_net.py, ...)
│       │   └── PartialReID/  (configs/, partialreid/, README.md, train_net.py)
│       ├── tests/
│       │   ├── __init__.py
│       │   ├── dataset_test.py
│       │   ├── feature_align.py
│       │   ├── interp_test.py
│       │   ├── lr_scheduler_test.py
│       │   ├── model_test.py
│       │   ├── sampler_test.py
│       │   └── test_repvgg.py
│       └── tools/
│           ├── plain_train_net.py
│           ├── train_net.py
│           └── deploy/
│               ├── caffe_export.py
│               ├── caffe_inference.py
│               ├── onnx_export.py
│               ├── onnx_inference.py
│               ├── pytorch_to_caffe.py
│               ├── README.md
│               ├── trt_calibrator.py
│               ├── trt_export.py
│               ├── trt_inference.py
│               └── Caffe/
│
├── dataset/
│   ├── __init__.py
│   └── dataset.py
│
├── datasets/                         ← MOT17 dataset root (git-ignored)
│
├── assets/
│   ├── ddmp_git.png
│   ├── diffmot_git.png
│   ├── teaser_git.png
│   ├── DiffMOT_DanceTrack.gif
│   └── DiffMOT_SportsMOT.gif
│
├── TrackEval/                        ← same structure as KALMAN-YOLO/TrackEval
│   ├── .gitignore / LICENSE / Readme.md / setup files
│   ├── data/
│   │   ├── gt/mot_challenge/
│   │   │   ├── seqmaps/MOT17-train.txt
│   │   │   └── MOT17-train/  (MOT17-02..13-FRCNN/ each with gt/ + seqinfo.ini)
│   │   └── trackers/mot_challenge/MOT17-train/
│   │       ├── BOTSORT_FASTREID/
│   │       │   ├── pedestrian_summary.txt
│   │       │   ├── pedestrian_detailed.csv
│   │       │   └── data/  (MOT17-02..13-FRCNN.txt)
│   │       └── BOTSORT_SWIN/
│   │           ├── pedestrian_summary.txt
│   │           ├── pedestrian_detailed.csv
│   │           └── data/  (MOT17-02..13-FRCNN.txt)
│   ├── docs/  (same format docs as KALMAN-YOLO/TrackEval/docs)
│   ├── scripts/  (run_mot_challenge.py, run_bdd.py, ...)
│   ├── tests/  (test_mot17.py, test_davis.py, ...)
│   └── trackeval/  (eval.py, plotting.py, utils.py, baselines/, datasets/, metrics/)
│
└── outputs/
    └── mot17/
        └── diffmot/
            ├── MOT17-02-FRCNN.txt
            ├── MOT17-04-FRCNN.txt
            ├── MOT17-05-FRCNN.txt
            ├── MOT17-09-FRCNN.txt
            ├── MOT17-10-FRCNN.txt
            ├── MOT17-11-FRCNN.txt
            ├── MOT17-13-FRCNN.txt
            ├── pedestrian_summary.txt
            ├── pedestrian_detailed.csv
            ├── pedestrian_plot.pdf
            ├── pedestrian_plot.png
            └── TERMINALRESULTS.txt
```

---

## `BOTSORT-FASTREID/`

BoTSORT tracker with FastReID SBS-S50 (ResNeSt50) appearance model.

```
BOTSORT-FASTREID/
├── botsort_mot.py                    ← main BoTSORT + FastReID tracking script
├── eval_only.py                      ← run TrackEval evaluation without re-tracking
├── TRACKING.txt
│
├── BoT-SORT/                         ← included BoT-SORT library
│   ├── .gitignore
│   ├── LICENSE
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.cfg
│   ├── setup.py
│   ├── yolov8x.pt                    ← YOLOv8x detection weights
│   │
│   ├── assets/
│   │   ├── BoT-SORT-MOT17-06.mov
│   │   ├── BoT-SORT-MOT17-14.mov
│   │   ├── BoT-SORT-MOT20-08.mov
│   │   └── Results_Bubbles.png
│   │
│   ├── pretrained/
│   │   ├── mot17_sbs_S50.pth         ← FastReID SBS-S50 ReID weights
│   │   └── osnet_x0_25_msmt17.pth
│   │
│   ├── tracker/
│   │   ├── bot_sort.py
│   │   ├── mc_bot_sort.py
│   │   ├── basetrack.py
│   │   ├── gmc.py
│   │   ├── kalman_filter.py
│   │   ├── matching.py
│   │   ├── GMC_files/
│   │   └── tracking_utils/
│   │       ├── evaluation.py
│   │       ├── io.py
│   │       └── timer.py
│   │
│   ├── fast_reid/                    ← FastReID library
│   │   ├── fast_reid_interfece.py    ← inference interface used by botsort_mot.py
│   │   ├── CHANGELOG.md
│   │   ├── GETTING_STARTED.md
│   │   ├── INSTALL.md
│   │   ├── LICENSE
│   │   ├── MODEL_ZOO.md
│   │   ├── README.md
│   │   ├── .gitignore
│   │   ├── .github/
│   │   │   ├── FastReID-Logo.png
│   │   │   ├── wechat_group.png
│   │   │   ├── ISSUE_TEMPLATE/
│   │   │   └── workflows/
│   │   ├── configs/
│   │   │   ├── Base-AGW.yml
│   │   │   ├── Base-bagtricks.yml
│   │   │   ├── Base-MGN.yml
│   │   │   ├── Base-SBS.yml
│   │   │   ├── DukeMTMC/
│   │   │   ├── Market1501/
│   │   │   ├── MOT17/
│   │   │   ├── MOT20/
│   │   │   ├── MSMT17/
│   │   │   ├── VehicleID/
│   │   │   ├── VeRi/
│   │   │   └── VERIWild/
│   │   ├── datasets/
│   │   │   └── generate_mot_patches.py
│   │   ├── demo/
│   │   │   ├── demo.py
│   │   │   ├── predictor.py
│   │   │   ├── visualize_result.py
│   │   │   ├── plot_roc_with_pickle.py
│   │   │   └── README.md
│   │   ├── docker/
│   │   │   ├── Dockerfile
│   │   │   └── README.md
│   │   ├── docs/
│   │   ├── fastreid/                 ← FastReID core package (same structure as DiffMOT external/fastreid/fastreid)
│   │   ├── projects/
│   │   ├── tests/
│   │   └── tools/
│   │
│   ├── tools/
│   │   ├── demo.py
│   │   ├── mc_demo.py
│   │   ├── mc_demo_yolov7.py
│   │   ├── track.py
│   │   ├── track_yolov7.py
│   │   ├── interpolation.py
│   │   ├── export_onnx.py
│   │   ├── mota.py
│   │   └── datasets/
│   │
│   ├── VideoCameraCorrection/
│   │   └── VideoCameraCorrection/
│   │
│   ├── yolov7/                       ← YOLOv7 integration (alternative detector)
│   │   ├── detect.py
│   │   ├── export.py
│   │   ├── hubconf.py
│   │   ├── LICENSE.md
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── test.py
│   │   ├── train.py
│   │   ├── train_aux.py
│   │   ├── cfg/
│   │   │   ├── baseline/  (yolor-csp.yaml, yolov3.yaml, yolov4-csp.yaml, ...)
│   │   │   ├── deploy/    (yolov7.yaml, yolov7x.yaml, yolov7-tiny.yaml, ...)
│   │   │   └── training/  (yolov7.yaml, yolov7x.yaml, yolov7-tiny.yaml, ...)
│   │   ├── data/
│   │   │   ├── coco.yaml
│   │   │   ├── hyp.scratch.custom.yaml
│   │   │   ├── hyp.scratch.p5.yaml
│   │   │   ├── hyp.scratch.p6.yaml
│   │   │   └── hyp.scratch.tiny.yaml
│   │   ├── figure/
│   │   │   ├── horses_prediction.jpg
│   │   │   ├── mask.png
│   │   │   ├── performance.png
│   │   │   └── pose.png
│   │   ├── inference/
│   │   │   └── images/
│   │   │       └── horses.jpg
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── common.py
│   │   │   ├── experimental.py
│   │   │   ├── export.py
│   │   │   └── yolo.py
│   │   ├── scripts/
│   │   │   └── get_coco.sh
│   │   ├── tools/
│   │   │   ├── instance.ipynb
│   │   │   ├── keypoint.ipynb
│   │   │   ├── reparameterization.ipynb
│   │   │   ├── visualization.ipynb
│   │   │   ├── YOLOv7onnx.ipynb
│   │   │   └── YOLOv7trt.ipynb
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── activations.py
│   │       ├── add_nms.py
│   │       ├── autoanchor.py
│   │       ├── datasets.py
│   │       ├── general.py
│   │       ├── google_utils.py
│   │       ├── loss.py
│   │       ├── metrics.py
│   │       ├── plots.py
│   │       ├── torch_utils.py
│   │       ├── aws/
│   │       │   ├── __init__.py
│   │       │   ├── mime.sh
│   │       │   ├── resume.py
│   │       │   └── userdata.sh
│   │       ├── google_app_engine/
│   │       │   ├── additional_requirements.txt
│   │       │   ├── app.yaml
│   │       │   └── Dockerfile
│   │       └── wandb_logging/
│   │           ├── __init__.py
│   │           ├── log_dataset.py
│   │           └── wandb_utils.py
│   │
│   └── yolox/                        ← YOLOX integration (alternative detector)
│
├── TrackEval/                        ← evaluation library
│   ├── .gitignore
│   ├── LICENSE
│   ├── Readme.md
│   ├── minimum_requirements.txt
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── setup.cfg
│   ├── setup.py
│   ├── data/
│   │   ├── gt/
│   │   │   └── mot_challenge/
│   │   │       ├── seqmaps/
│   │   │       │   └── MOT17-train.txt
│   │   │       └── MOT17-train/
│   │   │           ├── MOT17-02-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           ├── MOT17-04-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           ├── MOT17-05-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           ├── MOT17-09-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           ├── MOT17-10-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           ├── MOT17-11-FRCNN/  (gt/ + seqinfo.ini)
│   │   │           └── MOT17-13-FRCNN/  (gt/ + seqinfo.ini)
│   │   └── trackers/
│   │       └── mot_challenge/
│   │           └── MOT17-train/
│   │               ├── BOTSORT_FASTREID/
│   │               │   ├── pedestrian_summary.txt    ← ★ eval summary (HOTA/MOTA/IDF1)
│   │               │   ├── pedestrian_detailed.csv
│   │               │   └── data/
│   │               │       ├── MOT17-02-FRCNN.txt
│   │               │       ├── MOT17-04-FRCNN.txt
│   │               │       ├── MOT17-05-FRCNN.txt
│   │               │       ├── MOT17-09-FRCNN.txt
│   │               │       ├── MOT17-10-FRCNN.txt
│   │               │       ├── MOT17-11-FRCNN.txt
│   │               │       └── MOT17-13-FRCNN.txt
│   │               └── BOTSORT_SWIN/
│   │                   ├── pedestrian_summary.txt    ← ★ eval summary (Swin experiment)
│   │                   ├── pedestrian_detailed.csv
│   │                   └── data/
│   │                       ├── MOT17-02-FRCNN.txt
│   │                       ├── MOT17-04-FRCNN.txt
│   │                       ├── MOT17-05-FRCNN.txt
│   │                       ├── MOT17-09-FRCNN.txt
│   │                       ├── MOT17-10-FRCNN.txt
│   │                       ├── MOT17-11-FRCNN.txt
│   │                       └── MOT17-13-FRCNN.txt
│   ├── docs/
│   │   ├── BDD100k-format.txt
│   │   ├── DAVIS-format.txt
│   │   ├── KITTI-format.txt
│   │   ├── MOTChallenge-format.txt
│   │   ├── MOTS-format.txt
│   │   ├── TAO-format.txt
│   │   ├── YouTube-VIS-format.txt
│   │   ├── How_To/Add_a_new_metric.md
│   │   ├── MOTChallenge-Official/Readme.md
│   │   ├── OpenWorldTracking-Official/Readme.md
│   │   └── RobMOTS-Official/Readme.md
│   ├── scripts/
│   │   ├── comparison_plots.py
│   │   ├── python
│   │   ├── run_bdd.py
│   │   ├── run_burst.py
│   │   ├── run_burst_ow.py
│   │   ├── run_davis.py
│   │   ├── run_headtracking_challenge.py
│   │   ├── run_kitti.py
│   │   ├── run_kitti_mots.py
│   │   ├── run_mot_challenge.py
│   │   ├── run_mots_challenge.py
│   │   ├── run_person_path_22.py
│   │   ├── run_rob_mots.py
│   │   ├── run_tao.py
│   │   ├── run_tao_ow.py
│   │   └── run_youtube_vis.py
│   ├── tests/
│   │   ├── test_all_quick.py
│   │   ├── test_davis.py
│   │   ├── test_metrics.py
│   │   ├── test_mot17.py
│   │   └── test_mots.py
│   └── trackeval/
│       ├── __init__.py
│       ├── _timing.py
│       ├── eval.py
│       ├── plotting.py
│       ├── utils.py
│       ├── baselines/
│       │   ├── __init__.py
│       │   ├── baseline_utils.py
│       │   ├── non_overlap.py
│       │   ├── pascal_colormap.py
│       │   ├── stp.py
│       │   ├── thresholder.py
│       │   └── vizualize.py
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── _base_dataset.py
│       │   ├── bdd100k.py
│       │   ├── burst.py
│       │   ├── burst_ow.py
│       │   ├── davis.py
│       │   ├── head_tracking_challenge.py
│       │   ├── kitti_2d_box.py
│       │   ├── kitti_mots.py
│       │   ├── mot_challenge_2d_box.py
│       │   ├── mots_challenge.py
│       │   ├── person_path_22.py
│       │   ├── rob_mots.py
│       │   ├── rob_mots_classmap.py
│       │   ├── run_rob_mots.py
│       │   ├── tao.py
│       │   ├── tao_ow.py
│       │   ├── youtube_vis.py
│       │   └── burst_helpers/
│       │       ├── __init__.py
│       │       ├── burst_base.py
│       │       ├── burst_ow_base.py
│       │       ├── BURST_SPECIFIC_ISSUES.md
│       │       ├── convert_burst_format_to_tao_format.py
│       │       ├── format_converter.py
│       │       └── tao_categories.json
│       └── metrics/
│           ├── __init__.py
│           ├── _base_metric.py
│           ├── clear.py
│           ├── count.py
│           ├── hota.py
│           ├── identity.py
│           ├── ideucl.py
│           ├── j_and_f.py
│           ├── track_map.py
│           └── vace.py
│
└── BOTSORT-OUTPUT/
    ├── botsort_fastreid_tracking.mp4
    └── track_results/
        ├── MOT17-02-FRCNN.txt
        ├── MOT17-04-FRCNN.txt
        ├── MOT17-05-FRCNN.txt
        ├── MOT17-09-FRCNN.txt
        ├── MOT17-10-FRCNN.txt
        ├── MOT17-11-FRCNN.txt
        └── MOT17-13-FRCNN.txt
```

---

## `SWIN-TRANSFORMER/`

Holds output results from a Swin Transformer-based tracking experiment. No source code lives here.

```
SWIN-TRANSFORMER/
└── BOTSORT-OUTPUT/
    ├── botsort_swin_tracking.mp4
    └── track_results/
        ├── MOT17-02-FRCNN.txt
        ├── MOT17-04-FRCNN.txt
        ├── MOT17-05-FRCNN.txt
        ├── MOT17-09-FRCNN.txt
        ├── MOT17-10-FRCNN.txt
        ├── MOT17-11-FRCNN.txt
        ├── MOT17-13-FRCNN.txt
        ├── pedestrian_summary.txt
        ├── pedestrian_detailed.csv
        ├── pedestrian_plot.pdf
        └── pedestrian_plot.png
```

---

## Outputs summary

| Pipeline           | Tracking results (.txt)                                       | Eval summary (pedestrian_summary.txt)                                                                         |
|--------------------|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Kalman + YOLOv8    | `KALMAN-YOLO/yolokfoutputs/track_results/`                    | `KALMAN-YOLO/yolokfoutputs/track_results/pedestrian_summary.txt`                                              |
| DiffMOT            | `DiffMOT-main/outputs/mot17/diffmot/`                         | `DiffMOT-main/outputs/mot17/diffmot/pedestrian_summary.txt`                                                   |
| BoTSORT + FastReID | `BOTSORT-FASTREID/BOTSORT-OUTPUT/track_results/`              | `BOTSORT-FASTREID/TrackEval/data/trackers/mot_challenge/MOT17-train/BOTSORT_FASTREID/pedestrian_summary.txt`  |
| Swin Transformer   | `SWIN-TRANSFORMER/BOTSORT-OUTPUT/track_results/`              | `SWIN-TRANSFORMER/BOTSORT-OUTPUT/track_results/pedestrian_summary.txt`                                        |

---

## Key dependencies

| Pipeline           | Core packages                                                       |
|--------------------|---------------------------------------------------------------------|
| All pipelines      | `ultralytics` (YOLOv8 detection)                                    |
| Kalman             | custom Kalman Filter, `TrackEval`                                   |
| DiffMOT            | `torch`, `torchvision`, `einops`, `tqdm`, `TrackEval`               |
| BoTSORT + FastReID | `torch`, `torchvision`, `fastreid`, `timm` (ResNeSt backbone)       |

---

## Git-ignored items

- Model weights: `*.pt`, `*.pth`
- Datasets: `DiffMOT-main/datasets/`, `KALMAN-YOLO/MOT17/`
- Large output folders: `KALMAN-YOLO/yolokfoutputs/`, `SWIN-TRANSFORMER/BOTSORT-OUTPUT/`

---

## Quick-start commands

```bash
# Kalman baseline
cd KALMAN-YOLO
python yolomot.py

# DiffMOT
cd DiffMOT-main
python main.py --config configs/mot17_test.yaml

# BoTSORT + FastReID
cd BOTSORT-FASTREID
python botsort_mot.py

# Evaluate only (BoTSORT + FastReID)
cd BOTSORT-FASTREID
python eval_only.py
```
