# Multi-person ST-GCN++ inference (isolated version)

This is an additive implementation. It does not replace or modify `model.py`,
the original `STGCNPP` class, the existing configuration, or the checkpoint.

## What it adds

- Accepts one person's keypoints as `[T,17,3]`.
- Accepts synchronized multi-person keypoints as `[P,T,17,3]`, up to
  `max_persons` (10 by default).
- Keeps the original COCO-17, 100-frame Joint Motion preprocessing.
- Loads the same PYSKL-compatible checkpoint when `num_classes=60`.
- Returns a structured prediction containing a scene-level result and one
  result per real person, with class ID, action name, confidence, top-k
  predictions, and the full probability vector.

## Usage

```python
import numpy as np
from GCN.STGCNPP.model_multi_person import STGCNPPMultiPerson

model = STGCNPPMultiPerson(
    checkpoint_path="GCN/checkpoints/stgcnpp_jm_ntu60_xsub_hrnet.pth",
    num_classes=60,
    device="cuda",
    max_persons=10,
)

# Synchronized tracks from YOLO Pose:
# shape = [number_of_people, frames, 17, 3]
# last dimension is x, y, confidence.
keypoints = np.zeros((3, 100, 17, 3), dtype=np.float32)

# Optional: set the actual video dimensions as (width, height).
# The default is the original PreNormalize2D size: (1920, 1080).
# No image or video frame is passed to ST-GCN++; it receives keypoints only.
# Optional: preserve IDs emitted by your YOLO tracker.
tracker_ids = [12, 18, 25]
result = model.predict(keypoints, topk=5, person_ids=tracker_ids)

print(result["num_persons"])
print(result["scene"]["action"], result["scene"]["confidence"])

for person in result["persons"]:
    print(
        person["person_id"],
        person["action"],
        person["confidence"],
        person["top_predictions"],
    )
```

## Output structure

- `num_persons`: number of actual people supplied (padding slots excluded).
- `scene`: aggregate prediction across supplied people.
- `persons`: independent predictions produced by applying the existing
  classifier layer to each person's pooled backbone features. Each item
  includes `person_id` (the supplied tracker ID, or a 1-based fallback) and
  `slot_index` (the position in the input tensor).
- Each prediction includes `class_id`, `action`, `confidence`,
  `top_predictions`, and `probabilities`.

The default class names are loaded from the adjacent `ntu60_labels.txt`.
For a custom classifier, pass a matching `class_names` list and a checkpoint
whose classifier has the same number of classes.

## Important interpretation

The original checkpoint was trained with two person slots. The backbone in
this repository uses `DATA_BN_TYPE="VC"`, so its parameter shapes are not
tied to the number of person slots. The new class therefore supports a
10-person tensor without reshaping or retraining the checkpoint.

However, accepting 10 people is an architectural compatibility result, not a
claim that the original two-person checkpoint has been trained or validated
for crowded 10-person scenes. Per-person predictions and scene aggregation
should be evaluated on representative project videos before being treated as
production-quality results.
