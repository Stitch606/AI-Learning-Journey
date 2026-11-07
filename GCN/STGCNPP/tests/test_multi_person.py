import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure the repository root is importable when pytest selects this test folder.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from GCN.STGCNPP.model_multi_person import STGCNPPMultiPerson


CHECKPOINT = ROOT / "GCN" / "checkpoints" / "stgcnpp_jm_ntu60_xsub_hrnet.pth"


def test_prepare_input_packs_up_to_ten_people():
    model = STGCNPPMultiPerson(max_persons=10)
    keypoints = np.zeros((10, 100, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 0.8
    packed, count = model.prepare_input(keypoints, image_size=(640, 480))
    assert packed.shape == (1, 1, 10, 100, 17, 3)
    assert count == 10


def test_prepare_input_rejects_more_than_configured_limit():
    model = STGCNPPMultiPerson(max_persons=10)
    keypoints = np.zeros((11, 100, 17, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="max_persons=10"):
        model.prepare_input(keypoints)


def test_forward_preserves_person_axis_for_ten_people():
    model = STGCNPPMultiPerson(num_classes=60, max_persons=10)
    model.eval()
    batch = torch.randn(1, 1, 10, 100, 17, 3)
    with torch.inference_mode():
        probabilities = model(batch)
    assert probabilities.shape == (1, 10, 60)
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(1, 10),
        atol=1e-5,
    )


def test_checkpoint_loads_strictly_and_keeps_person_axis():
    assert CHECKPOINT.is_file(), f"Missing checkpoint: {CHECKPOINT}"
    model = STGCNPPMultiPerson(
        checkpoint_path=CHECKPOINT,
        num_classes=60,
        device="cpu",
        strict=True,
        max_persons=3,
    )
    model.eval()
    batch = torch.randn(1, 1, 3, 100, 17, 3)
    with torch.inference_mode():
        probabilities = model(batch)
    assert probabilities.shape == (1, 3, 60)
    assert torch.isfinite(probabilities).all()


def test_class_names_match_ntu60_checkpoint():
    model = STGCNPPMultiPerson()
    assert len(model.class_names) == 60
    assert model.class_names[42] == "falling"
