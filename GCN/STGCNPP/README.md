# Standalone PyTorch ST-GCN++ / COCO-17

This package provides a standalone PyTorch implementation of the ST-GCN++
Joint Motion (JM) model for YOLO COCO-17 pose sequences.

## Runtime dependencies
- PyTorch
- NumPy

The model implementation does not import or require PYSKL, MMCV, MMAction,
MMEngine, registries, or upstream config/runtime code.

## Public API

The only public model entry point is `STGCNPP`:

```python
from GCN.STGCNPP import STGCNPP

model = STGCNPP(
    checkpoint_path="GCN/checkpoints/stgcnpp_jm_ntu60_xsub_hrnet.pth",
    num_classes=60,
    device="cuda",
)

# keypoints: NumPy array with shape [T,17,3], in YOLO COCO-17 order.
probabilities = model.predict(keypoints, image_size=(1920, 1080))
class_id = probabilities.argmax(dim=-1).item()
```

The constructor loads the checkpoint and moves the model to the selected device.
`predict()` validates the raw keypoints, samples 100 frames, applies the JM
preprocessing, packs the two-person input, and returns class probabilities.
The lower-level `forward()` accepts an already-prepared tensor shaped
`[B, clips, persons, T, V, C]`.

## Architecture and checkpoint
The internal PyTorch modules preserve the backbone/head state-dict paths:
`backbone.*` and `cls_head.*`. The graph construction helpers belong to the
backbone; checkpoint loading and YOLO/JM input handling belong to `STGCNPP`.
These implementation details are not separate public entry points.

## Automated verification
GitHub Actions downloads the official JM checkpoint when it is not already
present, then tests COCO-17 graph order, input validation/preprocessing,
strict checkpoint loading, state-dict names, and a complete forward pass.
The standalone test workflow installs only PyTorch, NumPy, and pytest; it does
not install PYSKL or OpenMMLab packages.

The pretrained head contains 60 NTU RGB+D classes. Fine-tuning is required
when the target application uses a different action label set.
