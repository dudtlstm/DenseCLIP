# DenseCLIP/segmentation/tools/make_btcv_training10.py
import os
import random
import shutil

def make_training10(src_dir, dst_dir, ratio=0.1, seed=42):
    random.seed(seed)

    all_files = [f for f in os.listdir(src_dir) if f.endswith('.png')]
    total_files = len(all_files)
    num_to_copy = int(total_files * ratio)

    print(f"Found {total_files} PNG files in {src_dir}")
    print(f"Randomly selecting {num_to_copy} files to copy to {dst_dir}")

    selected_files = random.sample(all_files, num_to_copy)

    os.makedirs(dst_dir, exist_ok=True)

    for fname in selected_files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        shutil.copy2(src_path, dst_path)

    print("Copy completed.")

if __name__ == "__main__":
    src = "/home/ys1024/DenseCLIP/data/BTCV/images/training"
    dst = "/home/ys1024/DenseCLIP/data/BTCV/images/training10"

    make_training10(src, dst)