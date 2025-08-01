import os
import numpy as np
from scipy.sparse import load_npz
from mmseg.datasets.btcv import BTCVDataset


# (1) CLASSES 순서 확인
class_names = [
    'spleen', 'kidney_right', 'kidney_left', 'gallbladder',
    'esophagus', 'liver', 'stomach', 'aorta', 'inferior_vena_cava',
    'portal_vein_and_splenic_vein', 'pancreas',
    'adrenal_gland_right', 'adrenal_gland_left'
]

print("✅ Step 1: class_names vs BTCVDataset.CLASSES")
assert class_names == list(BTCVDataset.CLASSES), "❌ 클래스 순서 불일치!"
print("✔️ 클래스 순서 동일합니다.\n")

# (2) npz 내부에 각 클래스가 실제로 있는지 체크
print("✅ Step 2: .npz 내부 클래스 존재 여부 확인")

npz_dir = "/home/ys1024/DenseCLIP/data/BTCV/annotations/training"
sample_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
sample_files.sort()

channel_sums = np.zeros((14,))

for i, fname in enumerate(sample_files):
    path = os.path.join(npz_dir, fname)
    sparse = load_npz(path)
    dense = sparse.toarray()

    if dense.shape[0] == 13:
        dense = np.vstack([np.zeros((1, dense.shape[1])), dense])  # background 추가
    if dense.shape[0] != 14:
        print(f"❌ [{fname}] shape mismatch: {dense.shape}")
        continue

    channel_sums += dense.sum(axis=1)

print("Sum of each class over all .npz (training):")
for i, name in enumerate(class_names):
    print(f"{i:2d} {name:30s} sum={channel_sums[i]:.1f}")

# (3) 문제 클래스 있는지 확인
print("\n❗️ 0인 클래스가 있으면 그 클래스는 training에 등장하지 않은 것!")

# 13 class test ------------------------
# import os
# import numpy as np
# from scipy.sparse import load_npz
# from mmseg.datasets.btcv import BTCVDataset

# # background를 제외한 13개 클래스 이름
# class_names = [
#     'spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'esophagus',
#     'liver', 'stomach', 'aorta', 'inferior_vena_cava',
#     'portal_vein_and_splenic_vein', 'pancreas',
#     'adrenal_gland_right', 'adrenal_gland_left'
# ]

# print("✅ Step 1: BTCVDataset.CLASSES[1:] 과 비교")  # background 제외 비교
# assert class_names == list(BTCVDataset.CLASSES[1:]), "❌ 클래스 순서 불일치!"
# print("✔️ 클래스 순서 동일합니다.\n")

# # training npz 디렉토리
# npz_dir = "/home/ys1024/DenseCLIP/data/BTCV/annotations/validation"
# sample_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
# sample_files.sort()

# channel_sums = np.zeros((13,))

# for i, fname in enumerate(sample_files):
#     path = os.path.join(npz_dir, fname)
#     sparse = load_npz(path)
#     dense = sparse.toarray()

#     # dense는 보통 shape = (13, H×W)
#     if dense.shape[0] != 13:
#         print(f"❌ [{fname}] shape mismatch: {dense.shape}")
#         continue

#     channel_sums += dense.sum(axis=1)

# print("Sum of each class (foreground only) over all .npz (validation):")
# for i, name in enumerate(class_names):
#     print(f"{i+1:2d} {name:30s} sum={channel_sums[i]:.1f}")

# print("\n❗️ sum=0인 클래스가 있으면 validation에 등장하지 않은 것!")