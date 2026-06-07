import os
from collections import defaultdict

def convert(det_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    frames = defaultdict(list)

    with open(det_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            frame = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            score = float(parts[6]) if len(parts) > 6 else 1.0

            frames[frame].append(f"0,{x},{y},{w},{h},{score}\n")

    for frame, dets in frames.items():
        file_path = os.path.join(out_dir, f"{frame:06d}.txt")
        with open(file_path, "w") as f:
            f.writelines(dets)

    print("DONE:", out_dir)


# Config
base_dir = r"C:\Users\vansh\VANSHIKASOHAL\DiffMOT-main\datasets\MOT17\train"
seq = "MOT17-04-FRCNN"

det_file = os.path.join(base_dir, seq, "det", "det.txt")
out_dir = os.path.join(base_dir, seq, "det_framewise")

print("Looking for:", det_file)
print("Exists:", os.path.exists(det_file))

if os.path.exists(det_file):
    convert(det_file, out_dir)
else:
print("det.txt NOT FOUND:", det_file)