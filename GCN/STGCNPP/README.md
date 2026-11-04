# Official PYSKL ST-GCN++ / COCO-17 integration

This package replaces the previous custom `GCN/model.py` implementation with an
explicit adapter to the upstream PYSKL `RecognizerGCN + STGCN` implementation
and its published `stgcn++_ntu60_xsub_hrnet/jm.py` configuration.

## Selected upstream model
- Backbone: PYSKL `STGCN` (ST-GCN++)
- Graph: `layout='coco'`, `mode='spatial'` (17 joints)
- Input channels: 3
- Temporal feature: Joint Motion (`jm`)
- Clip length: 100 frames
- Person slots: 2
- Pretraining head: 60 NTU RGB+D classes

## YOLO compatibility
Ultralytics COCO-17 pose output uses the same 17-joint order encoded in
`COCO17_NAMES`. The adapter validates [T,17,3] (x,y,confidence) arrays and
packs them into the recognizer's [B,clips,M,T,V,C] contract.

The official PYSKL `PreNormalize2D` and `GenSkeFeat(dataset='coco',
feats=['jm'])` transforms must be applied to the raw coordinates before
inference. The adapter does not pretend raw coordinates are already JM features.

## Automated verification
GitHub Actions downloads the official checkpoint at runtime (it is not committed
to Git), loads the full recognizer with `strict=True`, and runs a forward pass.
The test also checks the COCO-17 joint count/order and YOLO tensor adapter.

Workflow: `.github/workflows/test-stgcnpp-coco17.yml`

A green workflow is the acceptance criterion for the architecture/checkpoint
load and model forward test. End-to-end action recognition still requires
fine-tuning the 60-class pretrained head on the project's target action labels.
