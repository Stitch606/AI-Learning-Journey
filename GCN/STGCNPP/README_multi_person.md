# Multi-person ST-GCN++ inference

This implementation provides independent action predictions for each tracked
person. It does not produce a scene-level action by averaging people.

## Input

- One person: `[T,17,3]`
- Multiple synchronized people: `[P,T,17,3]`
- Streaming batches: `[B,clips,T,P,17,3]`

The final dimension is YOLO COCO-17 `(x, y, confidence)`. Each person keeps
its own temporal keypoint sequence and tracker ID.

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

keypoints = np.zeros((3, 100, 17, 3), dtype=np.float32)
tracker_ids = [12, 18, 25]
result = model.predict(keypoints, topk=5, person_ids=tracker_ids)

for person in result["persons"]:
    print(person["person_id"], person["action"], person["confidence"])
```

## Output

- `num_persons`: number of people supplied.
- `persons`: one independent result per person, including `person_id`,
  `slot_index`, `class_id`, `action`, `confidence`, `top_predictions`,
  and the full probability vector.
- There is no `scene` result and no mean across people.

## Checkpoint compatibility

The model keeps the original backbone and classifier parameter names, so the
checkpoint weights and tensor shapes remain load-compatible. The backbone
configuration uses `DATA_BN_TYPE="VC"`; its learned normalization parameters
are not tied to a fixed person count. The shared backbone processes the
person axis, and the classifier applies the same learned `fc_cls` layer to
each person's pooled feature independently.

The original checkpoint's training path aggregates person features before
classification. This implementation changes inference to per-person
classification without retraining or changing weights. Consequently, strict
checkpoint loading is preserved, while per-person accuracy must be measured
on representative videos. A limit of 10 people is a supported input/runtime
capacity, not a claim that the checkpoint was trained on crowded 10-person
scenes.
