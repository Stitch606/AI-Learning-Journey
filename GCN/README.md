# ST-GCN++ — PYSKL Joint Motion / COCO-17

This module is a standalone PyTorch port of the **official PYSKL ST-GCN++**
architecture and the configuration used by its published 2D Joint-Motion
checkpoint:

- Config: `configs/stgcn++/stgcn++_ntu60_xsub_hrnet/jm.py`
- Dataset setting: NTU60 XSub, HRNet 2D skeleton annotations
- Graph: COCO-17, spatial partition
- Input channels: x, y, confidence
- Clip length: 100 frames
- Persons: 2 slots (the pipeline pads a single detected person)
- GCN: adaptive='init', with residual
- TCN: mstcn
- Backbone output: 256 channels
- Original classifier: 60 classes

## Files

- `model.py`: PYSKL-compatible ST-GCN++ backbone, multi-scale TCN, GCN,
  and GCN classification head. The module names follow PYSKL so the published
  checkpoint can be loaded with `load_pyskl_checkpoint()`.
- `graph/coco.py`: COCO-17 graph with the same inward/outward edge orientation
  and adjacency normalization as PYSKL.
- `config.py`: checkpoint and project configuration constants.

## Official Joint-Motion checkpoint

`https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/jm.pth`

PYSKL reports 84.0% top-1 on NTU60 XSub for this 2D HRNet Joint-Motion model.
The checkpoint's classifier has 60 NTU action classes. For a project-specific
classifier, load the pretrained backbone first and replace/fine-tune the head
for the project's labels.

## Input

Accepted tensors:

- `[N, T, 17, 3]` for one person
- `[N, M, T, 17, 3]` for multiple people

The model expects 2D coordinates and confidence, not XYZ depth. The published
checkpoint uses 100-frame clips and PYSKL's 2D normalization and Joint-Motion
feature generation. YOLO Pose uses the COCO-17 joint semantics/order, but its
coordinate normalization and confidence distribution must still be processed
consistently with the checkpoint pipeline.

## Example

```python
import torch
from GCN.model import STGCNPP

model = STGCNPP(num_classes=60)
model.load_pyskl_checkpoint("jm.pth")

keypoints = torch.randn(2, 100, 17, 3)  # N, T, V, C
logits = model(keypoints)
print(logits.shape)  # [2, 60]
```

Reference implementation and configuration:
https://github.com/kennymckormick/pyskl
