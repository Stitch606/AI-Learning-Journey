"""Thin, explicit adapter around the official PYSKL ST-GCN++ implementation.

The backbone/recognizer implementation is imported from the pinned upstream PYSKL
source in CI. This avoids silently maintaining a divergent reimplementation.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

PYSKL_COCO17_INWARD_EDGES = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
    (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
    (1, 0), (3, 1), (2, 0), (4, 2),
)


def validate_yolo_coco17(keypoints: np.ndarray) -> np.ndarray:
    """Validate YOLO COCO-17 data in [T,17,3] (x,y,confidence) format."""
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (17, 3):
        raise ValueError(
            f"Expected YOLO COCO-17 array [T,17,3], got {tuple(arr.shape)}"
        )
    if arr.shape[0] < 1:
        raise ValueError("At least one frame is required")
    if not np.isfinite(arr).all():
        raise ValueError("Keypoints contain NaN or infinite values")
    if np.any(arr[..., 2] < 0) or np.any(arr[..., 2] > 1):
        raise ValueError("YOLO keypoint confidence must be in [0,1]")
    return arr


def prepare_yolo_keypoints(keypoints: np.ndarray, max_persons: int = 2) -> np.ndarray:
    """Pack one YOLO person track into PYSKL recognizer input [1,1,M,T,17,3].

    This is the coordinate tensor adapter. The official JM feature transform
    (PreNormalize2D + GenSkeFeat(dataset='coco', feats=['jm'])) is applied by
    the PYSKL inference pipeline before passing data to the recognizer.
    """
    arr = validate_yolo_coco17(keypoints)
    if max_persons != 2:
        raise ValueError("The selected PYSKL checkpoint/config uses num_person=2")
    t = arr.shape[0]
    packed = np.zeros((1, 1, max_persons, t, 17, 3), dtype=np.float32)
    packed[0, 0, 0] = arr
    return packed


def build_stgcnpp(config_path: str | Path, checkpoint_path: str | Path, device: str = "cpu"):
    """Build the official PYSKL recognizer and load every checkpoint tensor strictly."""
    import torch
    from mmcv import Config
    from pyskl.models import build_model

    cfg = Config.fromfile(str(config_path))
    model = build_model(cfg.model)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state_dict mapping")
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cfg
