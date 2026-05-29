from torch.utils.data import Dataset
import numpy as np
import os
import glob


class DiffMOTDataset(Dataset):
    def __init__(self, path, config=None):

        path = os.path.abspath(path)

        self.config = config
        self.interval = self.config.interval + 1

        self.trackers = {}
        self.images = {}
        self.nframes = {}
        self.ntrackers = {}

        self.nsamples = {}
        self.nS = 0

        self.nds = {}
        self.cds = {}

        if os.path.isdir(path):

            if 'MOT' in path:
                self.seqs = [
                    "MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
                    "MOT17-10", "MOT17-11", "MOT17-13",
                    "MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"
                ]
            else:
                self.seqs = [
                    s for s in os.listdir(path)
                    if s.startswith("MOT17")
                ]

            self.seqs.sort()

            lastindex = 0

            for seq in self.seqs:

                trackerPath = os.path.abspath(
                    os.path.join(path, seq, "det", "det.txt")
                )

                if not os.path.exists(trackerPath):
                    print("SKIP MISSING:", trackerPath)
                    continue

                print("\nSEQ:", seq)
                print("PATH:", trackerPath)
                print("EXISTS:", os.path.exists(trackerPath))

                self.trackers[seq] = [trackerPath]
                self.ntrackers[seq] = len(self.trackers[seq])

                imagePath = os.path.join(path, seq, "img1", "*.*")
                self.images[seq] = sorted(glob.glob(imagePath))
                self.nframes[seq] = len(self.images[seq])

                self.nsamples[seq] = {}

                for i, pa in enumerate(self.trackers[seq]):

                    pa = os.path.abspath(pa)

                    if not os.path.exists(pa):
                        print("MISSING FILE:", pa)
                        continue

                    try:
                        data = np.loadtxt(pa, delimiter=",")
                    except Exception as e:
                        print("BAD FILE:", pa, "ERROR:", e)
                        continue

                    if len(data.shape) == 1:
                        data = data.reshape(1, -1)

                    if len(data) == 0:
                        continue

                    self.nsamples[seq][i] = len(data) - self.interval
                    self.nS += self.nsamples[seq][i]

                if len(self.nsamples[seq]) == 0:
                    print("NO VALID TRACKERS:", seq)
                    continue

                self.nds[seq] = list(self.nsamples[seq].values())
                self.cds[seq] = [
                    sum(self.nds[seq][:i]) + lastindex
                    for i in range(len(self.nds[seq]))
                ]

                print("TRACKERS:", len(self.trackers[seq]))
                print("IMAGES:", len(self.images[seq]))

                lastindex = self.cds[seq][-1] + self.nds[seq][-1]

        print("=" * 80)
        print("dataset summary:", self.nS)
        print("=" * 80)

    def __getitem__(self, files_index):

        ds, trk, start_index = None, None, None

        for seq in self.cds:
            if files_index >= self.cds[seq][0]:
                ds = seq
                for j, c in enumerate(self.cds[seq]):
                    if files_index >= c:
                        trk = j
                        start_index = c
                    else:
                        break
            else:
                break

        track_path = self.trackers[ds][trk]
        track_gt = np.loadtxt(track_path, dtype=np.float32, delimiter=",")

        init_index = files_index - start_index
        cur_index = init_index + self.interval

        cur_gt = track_gt[cur_index]
        cur_bbox = cur_gt[2:6]

        boxes = [
            track_gt[init_index + i][2:6]
            for i in range(self.interval)
        ]

        delt_boxes = [
            boxes[i + 1] - boxes[i]
            for i in range(self.interval - 1)
        ]

        conds = np.concatenate(
            (np.array(boxes)[1:], np.array(delt_boxes)),
            axis=1
        )

        delt = cur_bbox - boxes[-1]

        return {
            "cur_gt": cur_gt,
            "cur_bbox": cur_bbox,
            "condition": conds,
            "delta_bbox": delt
        }

    def __len__(self):
        return self.nS