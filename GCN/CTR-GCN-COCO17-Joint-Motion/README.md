# CTR-GCN — COCO-17 Joint Motion

This folder contains the PySKL CTR-GCN Joint Motion configuration using the COCO skeleton layout (17 keypoints).

## Model details
- Architecture: CTR-GCN
- Input layout: COCO-17 (graph_cfg.layout='coco')
- Modality: Joint Motion (GenSkeFeat with feats=['jm'])
- Input type: 2D HRNet skeletons
- Training/evaluation dataset: NTU RGB+D 60, cross-subject split
- Reported top-1 accuracy: 89.4% on NTU60 XSub (PySKL model zoo)

Important: COCO-17 describes the keypoint topology. The action classes and reported accuracy are from NTU60, not the COCO dataset. The pipeline performs 2D normalization, joint-motion feature generation, temporal sampling to 100 frames, and formats up to 2 people.

## Files
- config.py: model and preprocessing configuration copied from PySKL.
- download_weights.py: downloads the official pretrained checkpoint into this folder.
- weights/ctrgcn_coco17_joint_motion.pth: created by running the downloader.

## Download pretrained weights
From the repository root:
python GCN/CTR-GCN-COCO17-Joint-Motion/download_weights.py

Official checkpoint:
https://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/jm.pth

Official source config:
https://github.com/kennymckormick/pyskl/blob/main/configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/jm.py

## Compatibility note
The config uses the COCO-17 graph and Joint Motion preprocessing. Feed keypoints in COCO-17 joint order and match PySKL coordinate normalization and tensor formatting. Raw YOLO pose output must pass through the same preprocessing pipeline; do not pass it to the network as-is.
