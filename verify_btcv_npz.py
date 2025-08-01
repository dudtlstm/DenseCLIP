import os
import numpy as np
from scipy.sparse import load_npz

root = "/home/ys1024/DenseCLIP/data/BTCV/annotations/training"
invalid_files = []

for fname in sorted(os.listdir(root)):
    if not fname.endswith(".npz"):
        continue
    path = os.path.join(root, fname)
    try:
        sparse = load_npz(path)
        arr = sparse.toarray()

        if arr.shape != (13, 512 * 512):
            print(f"[❌ INVALID SHAPE] {fname}: {arr.shape}")
            invalid_files.append(path)

    except Exception as e:
        print(f"[💥 ERROR] {fname}: {e}")
        invalid_files.append(path)

print("\n[SUMMARY]")
print(f"Total files checked: {len(os.listdir(root))}")
print(f"Invalid .npz files: {len(invalid_files)}")