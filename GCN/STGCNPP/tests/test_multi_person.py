import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure the repository root is importable when pytest selects this test folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from GCN.STGCNPP.model_multi_person import STGCNPPMultiPerson


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


def test_forward_accepts_ten_person_dimension():
    model = STGCNPPMultiPerson(num_classes=60, max_persons=10)
    model.eval()
    batch = torch.zeros(1, 1, 10, 100, 17, 3)
    with torch.inference_mode():
        probabilities = model(batch)
    assert probabilities.shape == (1, 10, 60)
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(1, 10),
        atol=1e-5,
    )


def test_predict_returns_only_independent_person_results():
    model = STGCNPPMultiPerson(num_classes=60, max_persons=10)
    keypoints = np.zeros((3, 100, 17, 3), dtype=np.float32)
    keypoints[..., 2] = 0.8
    result = model.predict(keypoints, person_ids=[12, 18, 25])
    assert result["num_persons"] == 3
    assert "scene" not in result
    assert [person["person_id"] for person in result["persons"]] == [12, 18, 25]
    assert len(result["persons"]) == 3


def test_class_names_match_ntu60_checkpoint():
    model = STGCNPPMultiPerson()
    assert len(model.class_names) == 60
    assert model.class_names[42] == "falling"
