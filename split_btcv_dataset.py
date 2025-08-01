import json
import os
import shutil

# 경로 설정
base_dir = "/home/ys1024/DenseCLIP/data/BTCV"
json_path = os.path.join(base_dir, "dataset.json")

# 타겟 폴더
image_train_dst = os.path.join(base_dir, "images", "training")
image_val_dst = os.path.join(base_dir, "images", "validation")
label_train_dst = os.path.join(base_dir, "annotations", "training")
label_val_dst = os.path.join(base_dir, "annotations", "validation")

# 디렉토리 생성
os.makedirs(image_train_dst, exist_ok=True)
os.makedirs(image_val_dst, exist_ok=True)
os.makedirs(label_train_dst, exist_ok=True)
os.makedirs(label_val_dst, exist_ok=True)

# JSON 로드
with open(json_path, "r") as f:
    data = json.load(f)

# 파일 이동 함수
def move_entry(entry, is_val):
    # 이미지 이동
    src_img = os.path.join(base_dir, entry["image"])
    dst_img = os.path.join(image_val_dst if is_val else image_train_dst, os.path.basename(entry["image"]))

    if os.path.exists(src_img):
        shutil.move(src_img, dst_img)
    else:
        print(f"🚫 이미지 없음: {src_img}")

    # 라벨 이동
    src_lbl = os.path.join(base_dir, entry["label"])
    dst_lbl = os.path.join(label_val_dst if is_val else label_train_dst, os.path.basename(entry["label"]))

    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)
    else:
        print(f"🚫 라벨 없음: {src_lbl}")

# validation
for entry in data["test"]:
    move_entry(entry, is_val=True)

# training
for entry in data["training"]:
    move_entry(entry, is_val=False)

print("✅ image와 label 파일 분할 완료!")