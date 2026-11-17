# Model assets

This folder contains the original NTU60 ST-GCN++ checkpoint and the generated browser model files.

- `stgcnpp_ntu60_xsub_hrnet.pth`: copied original 5.85 MB checkpoint.
- `stgcnpp_ntu60.onnx`: generated from the original two-person architecture during deployment.
- `ntu60_labels.txt`: the original 60 NTU RGB+D class labels.
- `model.param`, `model.bin`, `model-config.json`: generated YOLO26n-Pose NCNN FP16 model files.

The ONNX export is checked against PyTorch in a separate workflow. The browser smoke test then runs actual ONNX Runtime Web inference using a two-person, 100-frame input.
