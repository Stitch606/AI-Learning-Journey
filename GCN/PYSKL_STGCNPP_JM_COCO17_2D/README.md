# PYSKL ST-GCN++ — 2D COCO-17 Joint Motion

This folder tracks the official PYSKL checkpoint/configuration selected for the project.

## Selected checkpoint

- Architecture family: ST-GCN++ (PYSKL implementation)
- Modality: Joint Motion (`jm`)
- Input annotation: HRNet 2D skeleton
- Graph layout: `coco` (COCO-17)
- Clip length: 100 frames
- Max persons in the official pipeline: 2
- Pretraining dataset: NTU RGB+D XSub, 60 action classes

## Files

- `jm.py`: PYSKL model and preprocessing configuration.
- `download_weights.py`: downloads the official binary checkpoint as `jm.pth`.
- `jm.pth`: downloaded checkpoint (not included in Git until the download workflow is run).

## Download locally

```bash
python GCN/PYSKL_STGCNPP_JM_COCO17_2D/download_weights.py
```

Official checkpoint:
https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/jm.pth

Official config:
https://github.com/kennymckormick/pyskl/blob/main/configs/stgcn%2B%2B/stgcn%2B%2B_ntu60_xsub_hrnet/jm.py

## Important compatibility note

The config uses PYSKL's own `RecognizerGCN` + `STGCN` implementation and its matching state-dict layout. The repository's existing `GCN/model.py` is a separate custom implementation. Do not assume this checkpoint can be loaded into that custom class with `load_state_dict`; use the PYSKL config/model implementation or explicitly port and map the checkpoint weights.

The checkpoint is pretrained on NTU RGB+D action labels (60 classes), not on this project's custom action labels. Fine-tuning is required for the project's classes.
