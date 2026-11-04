from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from GCN.STGCNPP.model import (
    COCO17_NAMES, PYSKL_COCO17_INWARD_EDGES,
    build_stgcnpp, prepare_yolo_jm, validate_yolo_coco17,
)

PYSKL_ROOT = Path("/tmp/pyskl")
CONFIG = PYSKL_ROOT / "configs/stgcn++/stgcn++_ntu60_xsub_hrnet/jm.py"
CHECKPOINT = Path("GCN/checkpoints/stgcnpp_jm_coco17.pth")


def test_coco17_layout_matches_pyskl_graph():
    assert len(COCO17_NAMES) == 17
    assert len(PYSKL_COCO17_INWARD_EDGES) == 16
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in PYSKL_COCO17_INWARD_EDGES)
    from pyskl.utils.graph import Graph
    official_graph = Graph(layout="coco", mode="spatial")
    assert official_graph.num_node == 17
    assert tuple(official_graph.inward) == PYSKL_COCO17_INWARD_EDGES
    # COCO-17 ordering used by Ultralytics YOLO pose and PYSKL's coco Graph.
    assert COCO17_NAMES == (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )


def test_yolo_to_official_joint_motion_shape_and_values():
    sample = np.zeros((100, 17, 3), dtype=np.float32)
    sample[..., 2] = 1.0
    packed = prepare_yolo_jm(sample, image_size=(1920, 1080), clip_len=100)
    assert packed.shape == (1, 1, 2, 100, 17, 3)
    assert np.isfinite(packed).all()
    assert np.count_nonzero(packed[0, 0, 1]) == 0


def test_reject_wrong_joint_count_and_bad_confidence():
    with np.testing.assert_raises(ValueError):
        validate_yolo_coco17(np.zeros((10, 16, 3), dtype=np.float32))
    bad = np.zeros((10, 17, 3), dtype=np.float32)
    bad[0, 0, 2] = 1.2
    with np.testing.assert_raises(ValueError):
        validate_yolo_coco17(bad)


def test_official_checkpoint_strict_load_and_forward():
    assert CONFIG.is_file(), f"Missing upstream PYSKL config: {CONFIG}"
    assert CHECKPOINT.is_file(), f"Missing downloaded checkpoint: {CHECKPOINT}"
    model, cfg = build_stgcnpp(CONFIG, CHECKPOINT, device="cpu")
    assert cfg.model.backbone.graph_cfg.layout == "coco"
    assert cfg.model.backbone.get('in_channels', 3) == 3
    assert cfg.model.cls_head.num_classes == 60

    # Exercise the complete official RecognizerGCN forward path using the
    # expected recognizer tensor contract: [B, clips, persons, T, V, C].
    raw_yolo = np.zeros((100, 17, 3), dtype=np.float32)
    raw_yolo[..., 0] = np.linspace(400, 600, 100)[:, None]
    raw_yolo[..., 1] = np.linspace(300, 500, 100)[:, None]
    raw_yolo[..., 2] = 0.95
    keypoint = torch.from_numpy(prepare_yolo_jm(raw_yolo))
    with torch.no_grad():
        scores = model(keypoint=keypoint, return_loss=False)
    assert scores.shape == (1, 60), f"Unexpected classifier output: {scores.shape}"
    assert torch.isfinite(torch.as_tensor(scores)).all()
