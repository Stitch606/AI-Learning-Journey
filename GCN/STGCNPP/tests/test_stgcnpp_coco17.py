from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from GCN.STGCNPP.model import (
    COCO17_NAMES, COCO17_INWARD, STGCNPP, build_stgcnpp,
    load_checkpoint, prepare_yolo_jm, validate_yolo_coco17,
)

CHECKPOINT = ROOT / "GCN/checkpoints/stgcnpp_jm_ntu60_xsub_hrnet.pth"


def test_standalone_has_no_upstream_runtime_imports():
    import inspect
    import GCN.STGCNPP.model as model_module
    source = inspect.getsource(model_module)
    for forbidden in ("import pyskl", "from pyskl", "import mmcv", "from mmcv",
                      "mmaction", "mmengine"):
        assert forbidden not in source
    assert len(COCO17_NAMES) == 17
    assert len(COCO17_INWARD) == 16


def test_checkpoint_strict_load_and_full_forward():
    assert CHECKPOINT.is_file(), f"Missing repository checkpoint: {CHECKPOINT}"
    model = STGCNPP(num_classes=60)
    result = load_checkpoint(model, CHECKPOINT, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    model.eval()

    # Non-degenerate moving skeleton, in pixel coordinates.
    raw = np.zeros((100, 17, 3), dtype=np.float32)
    for frame in range(100):
        raw[frame, :, 0] = 700 + frame * 1.5 + np.arange(17) * 8
        raw[frame, :, 1] = 250 + frame * 0.7 + np.arange(17) * 5
        raw[frame, :, 2] = 0.95
    sample = torch.from_numpy(prepare_yolo_jm(raw))
    with torch.no_grad():
        probabilities = model(sample)
    assert probabilities.shape == (1, 60), probabilities.shape
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(-1), torch.ones(1), atol=1e-5)


def test_yolo_coco17_validation_and_preprocessing():
    raw = np.zeros((25, 17, 3), dtype=np.float32)
    raw[..., 2] = 0.9
    packed = prepare_yolo_jm(raw)
    assert packed.shape == (1, 1, 2, 100, 17, 3)
    assert np.isfinite(packed).all()
    assert np.count_nonzero(packed[0, 0, 1]) == 0
    with np.testing.assert_raises(ValueError):
        validate_yolo_coco17(np.zeros((10, 16, 3), dtype=np.float32))
    bad = raw.copy()
    bad[0, 0, 2] = 1.2
    with np.testing.assert_raises(ValueError):
        validate_yolo_coco17(bad)


def test_state_dict_names_match_reference_contract():
    model = STGCNPP()
    keys = model.state_dict().keys()
    assert "backbone.data_bn.weight" in keys
    assert "backbone.gcn.0.gcn.A" in keys
    assert "backbone.gcn.0.tcn.branches.0.0.weight" in keys
    assert "cls_head.fc_cls.weight" in keys
