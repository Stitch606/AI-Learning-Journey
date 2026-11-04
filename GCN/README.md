# ST-GCN++ (COCO-17)

This directory contains a PyTorch implementation of the ST-GCN++-style
multi-scale spatial-temporal graph convolution network, configured by default
for the 17 COCO keypoints produced by YOLO Pose.

## Files
- `model.py`: model, spatial graph convolution, multi-scale temporal convolution,
  and classification head.
- `graph/coco.py`: COCO-17 skeleton topology and adjacency partitions.
- `config.py`: project defaults for sequence/model dimensions.

## Input
Preferred input shape: `[N, C, T, V, M]`.
For a single person, `[N, T, V, C]` is also accepted.
Default values: `C=3` (x, y, confidence), `V=17`, `M=1`.

## Example
```python
import torch
from GCN.model import STGCNPP

model = STGCNPP(num_classes=10, in_channels=3, num_person=1)
keypoints = torch.randn(2, 100, 17, 3)  # N, T, V, C
logits = model(keypoints)
print(logits.shape)  # [2, 10]
```

This code defines the architecture and initializes trainable parameters.
It does not claim that an NTU-60/NTU-120 checkpoint is directly compatible
with COCO-17. Checkpoint compatibility depends on joint count, joint order,
input channels, graph layout, and model configuration. Train on COCO-17 data
or use a checkpoint explicitly trained with the same layout.

Architecture reference: [PYSKL](https://github.com/kennymckormick/pyskl),
especially its ST-GCN backbone and multi-scale temporal convolution utilities.
This repository implementation is standalone and requires PyTorch only.
