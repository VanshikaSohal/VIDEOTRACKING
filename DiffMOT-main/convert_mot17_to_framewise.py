import os

def convert(det_path, out_path):
    os.makedirs(out_path, exist_ok=True)

    data = []

    # read MOT17 det.txt
    with open(det_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            frame = int(parts[0])
            x, y, w, h = parts[2:6]

            data.append((frame, x, y, w, h))

    # group by frame
    from collections import defaultdict
    frames = defaultdict(list)

    for frame, x, y, w, h in data:
        frames[frame].append(f"0,{x},{y},{w},{h},1\n")

    # write frame-wise files
    for frame in frames:
        file_name = os.path.join(out_path, f"{frame:06d}.txt")
        with open(file_name, "w") as f:
            f.writelines(frames[frame])

    print("DONE:", out_path)


# CHANGE THIS PATH
base_dir = r"C:\Users\vansh\VANSHIKASOHAL\DiffMOT-main\datasets\MOT17\train"

sequences = os.listdir(base_dir)

for seq in sequences:
    det_file = os.path.join(base_dir, seq, "det", "det.txt")

    if os.path.exists(det_file):
        out_dir = os.path.join(base_dir, seq, "det_framewise")
        convert(det_file, out_dir)