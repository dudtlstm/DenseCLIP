import os
import re

# 수정할 디렉토리 경로
base_dir = "/home/ys1024/DenseCLIP/data/BTCV/annotations"
subfolders = ['training', 'validation']

pattern = re.compile(r"x___(ABD_\d+_\d+)\.\(13, 512, 512, 1\)\.npz")

for split in subfolders:
    dir_path = os.path.join(base_dir, split)
    for fname in os.listdir(dir_path):
        if fname.endswith(".npz") and fname.startswith("x___"):
            match = pattern.match(fname)
            if match:
                new_name = match.group(1) + ".npz"
                src = os.path.join(dir_path, fname)
                dst = os.path.join(dir_path, new_name)
                os.rename(src, dst)
                print(f"✅ Renamed: {fname} → {new_name}")
            else:
                print(f"⚠️ Skipped: {fname} (pattern not matched)") # 매칭이 안 되면 skip하도록
