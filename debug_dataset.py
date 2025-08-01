from mmcv import Config
from mmseg.datasets import build_dataset
import segmentation.mmseg.datasets.btcv
import segmentation.mmseg.datasets.pipelines.load_btcv_ann
import numpy as np

cfg = Config.fromfile('segmentation/configs/denseclip_fpn_res50_512x512_80k_btcv.py')
dataset = build_dataset(cfg.data.train)

print(f"[INFO] Dataset length: {len(dataset)}")
invalid_count = 0

for i in range(len(dataset)):
    try:
        sample = dataset[i]
        img = sample['img'].data
        gt_seg = sample['gt_semantic_seg'].data

        # 조건: 정상인데도 문제 있는 경우도 출력하고 싶다면 아래 주석 해제
        # if gt_seg.shape != (512, 512):
        #     print(f"[WARNING] Abnormal shape at index {i}: {gt_seg.shape}")

    except Exception as e:
        print(f"[❌ EXCEPTION] at index {i}: {e}")
        invalid_count += 1

print(f"\n[SUMMARY] Total invalid samples: {invalid_count} / {len(dataset)}")
