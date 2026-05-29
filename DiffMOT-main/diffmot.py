import os
import torch
import numpy as np
import os.path as osp
import logging
import time
import cv2

from torch import optim, utils
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

from dataset import DiffMOTDataset
from models.autoencoder import D2MP
from models.condition_embedding import History_motion_embedding

from tracker.DiffMOTtracker import diffmottracker
from tracking_utils.log import logger
from tracking_utils.timer import Timer


def write_results(filename, results, data_type='mot'):
    if data_type == 'mot':
        fmt = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        fmt = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError("Unknown format")

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, tid in zip(tlwhs, track_ids):
                if tid < 0:
                    continue

                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h

                f.write(fmt.format(
                    frame=frame_id,
                    id=tid,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    w=w, h=h
                ))

    logger.info(f"Saved results to {filename}")


def mkdirs(path):
    os.makedirs(path, exist_ok=True)


class DiffMOT:
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self._build()

    # ===================== TRAIN =====================
    def train(self):
        for epoch in range(1, self.config.epochs + 1):

            self.train_dataset.augment = self.config.augment
            loader = self.train_data_loader

            for batch in tqdm(loader, ncols=80):
                for k in batch:
                    batch[k] = batch[k].cuda(non_blocking=True)

                loss = self.model(batch).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % self.config.eval_every == 0:
                ckpt = {
                    "ddpm": self.model.state_dict(),
                    "epoch": epoch,
                    "optimizer": self.optimizer.state_dict()
                }

                save_path = osp.join(
                    self.model_dir,
                    f"{self.config.dataset}_epoch{epoch}.pt"
                )
                torch.save(ckpt, save_path)
                print(f"[CHECKPOINT SAVED] {save_path}")

    # ===================== EVAL (FIXED) =====================
    def eval(self):

        det_root = self.config.det_dir
        if not os.path.exists(det_root):
            raise FileNotFoundError(f"Detections folder not found: {det_root}")

        img_root = getattr(self.config, "img_dir", None)

        seqs = sorted([
            s for s in os.listdir(det_root)
            if os.path.isdir(osp.join(det_root, s)) and not s.startswith('.')
        ])

        for seq in seqs:
            print(f"\nProcessing sequence: {seq}")

            det_path = osp.join(det_root, seq, "det_framewise")

            info_path = osp.join(self.config.info_dir, seq, "seqinfo.ini")
            if not os.path.exists(info_path):
                raise FileNotFoundError(info_path)

            seq_info = open(info_path).read()
            w = int(seq_info.split("imWidth=")[1].split("\n")[0])
            h = int(seq_info.split("imHeight=")[1].split("\n")[0])

            tracker = diffmottracker(self.config)
            timer = Timer()

            results = []

            frame_files = sorted([
                f for f in os.listdir(det_path)
                if f.endswith(".txt")
            ])

            if len(frame_files) == 0:
                print("No detection files found in:", det_path)
                continue

            for frame_id, f in enumerate(frame_files):

                det_file = osp.join(det_path, f)

                dets_raw = np.loadtxt(det_file, delimiter=',', dtype=np.float32)

                if dets_raw.size == 0:
                    continue

                if dets_raw.ndim == 1:
                    dets_raw = dets_raw[None, :]

                # x,y,w,h,score
                dets = dets_raw[:, 1:6]

                # ================= IMAGE FIX =================
                img = None
                if img_root is not None:
                    try:
                        frame_num = int(os.path.splitext(f)[0].split("_")[-1])
                        img_name = f"{frame_num:06d}.jpg"
                        img_path = osp.join(img_root, seq, "img1", img_name)

                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                    except:
                        img = None

                # fallback (prevents crash)
                if img is None:
                    img = np.zeros((h, w, 3), dtype=np.uint8)

                timer.tic()

                tag = f"{seq}:{frame_id + 1}"

                online_targets = tracker.update(
                    dets,
                    self.model,
                    frame_id,
                    w,
                    h,
                    tag,
                    img
                )

                timer.toc()

                tlwhs, ids = [], []
                for t in online_targets:
                    tlwhs.append(t.tlwh)
                    ids.append(t.track_id)

                results.append((frame_id + 1, tlwhs, ids))

            tracker.dump_cache()

            out_dir = self.config.save_dir
            mkdirs(out_dir)

            out_file = osp.join(out_dir, f"{seq}.txt")
            write_results(out_file, results)

            print(f"[DONE] {seq} saved")

    # ===================== BUILD =====================
    def _build(self):
        self._build_dir()
        self._build_encoder()
        self._build_model()

        if not self.config.eval_mode:
            self._build_train_loader()

        self._build_optimizer()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments", self.config.eval_expname)
        os.makedirs(self.model_dir, exist_ok=True)

        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder(self):
        self.encoder = History_motion_embedding()

    def _build_model(self):
        model = D2MP(self.config, encoder=self.encoder)
        self.model = model.cuda()

        if self.config.eval_mode:
            epoch = self.config.eval_at
            ckpt_path = osp.join(
                self.model_dir,
                f"{self.config.dataset}_epoch{epoch}.pt"
            )

            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(ckpt_path)

            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(
                {k.replace("module.", ""): v for k, v in ckpt["ddpm"].items()}
            )
            self.model.eval()

        print("> Model built!")

    def _build_train_loader(self):
        self.train_dataset = DiffMOTDataset(self.config.data_dir, self.config)

        self.train_data_loader = utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.preprocess_workers,
            pin_memory=True
        )

        print("> Train loader built!")