# ST-GCN++ — PYSKL Joint Motion / COCO-17

The model in this folder is a standalone PyTorch port of the official PYSKL
ST-GCN++ architecture and the configuration used by the published 2D
Joint-Motion checkpoint.

## Exact pretrained configuration

- Official config: `configs/stgcn++/stgcn++_ntu60_xsub_hrnet/jm.py`
- Dataset: NTU60 XSub, HRNet 2D skeleton annotations
- Graph: COCO-17, PYSKL spatial partition and edge orientation
- Input features: Joint Motion (JM), 3 channels
- Clip length: 100 frames
- Person slots: 2 (a single person is zero-padded to two slots)
- GCN: adaptive='init', with residual
- TCN: mstcn
- Backbone output: 256 channels
- Original classifier: 60 NTU action classes

## Files

- `model.py`: PYSKL-compatible ST-GCN++ backbone, GCN, multi-scale TCN,
  GCN head, checkpoint loader, and YOLO-to-JM preprocessing helper.
- `graph/coco.py`: exact COCO-17 adjacency orientation used by PYSKL.
- `config.py`: model and checkpoint configuration constants.

The model uses PYSKL-compatible state-dict names:
`backbone.data_bn`, `backbone.gcn.*`, and `cls_head.fc_cls`.

## Official Joint-Motion checkpoint

`https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/jm.pth`

PYSKL reports 84.0% top-1 accuracy on NTU60 XSub for this 2D HRNet
Joint-Motion model.

The published classifier has 60 NTU classes. To load the full checkpoint,
instantiate `STGCNPP(num_classes=60)`. For a project-specific classifier,
instantiate it with your class count and call
`load_pyskl_checkpoint(path, load_head=False)`; this loads the pretrained
backbone while leaving your new classifier initialized for fine-tuning.

## YOLO Pose input and preprocessing

YOLO Pose's COCO-17 joint order matches the PYSKL COCO-17 layout:

1. nose
2. left eye
3. right eye
4. left ear
5. right ear
6. left shoulder
7. right shoulder
8. left elbow
9. right elbow
10. left wrist
11. right wrist
12. left hip
13. right hip
14. left knee
15. right knee
16. left ankle
17. right ankle

The checkpoint expects 2D `x, y, confidence`, not XYZ depth. Its JM pipeline
first normalizes pixel coordinates using the source frame dimensions, masks
low-confidence coordinates, then computes frame-to-frame joint motion.
Use the included helper for that transformation and uniformly sample each
clip to 100 frames before inference.

```python
import torch
from GCN.model import STGCNPP, prepare_joint_motion

# 10 project-specific action classes:
model = STGCNPP(num_classes=10)
model.load_pyskl_checkpoint("jm.pth", load_head=False)

# YOLO keypoints in pixel coordinates: [N, T, 17, 3]
yolo_keypoints = torch.randn(2, 100, 17, 3)
jm = prepare_joint_motion(yolo_keypoints, image_size=(1080, 1920))
logits = model(jm)
print(logits.shape)  # [2, 10]
```

Reference architecture and configuration:
https://github.com/kennymckormick/pyskl
