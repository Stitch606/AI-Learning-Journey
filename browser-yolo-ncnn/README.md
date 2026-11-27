# Browser YOLO26 Pose + ST-GCN++ Action Recognition

Browser-local pipeline hosted as static files on GitHub Pages. Camera frames remain on the visitor's device.

## Models and precision

- YOLO26n-Pose is exported to ONNX at a fixed 512×512 input. The deployment build produces an FP16 ONNX model for WebGPU and a separate FP32 compatibility model for browsers without WebGPU.
- ST-GCN++ preserves the original two-person NTU60 XSub HRNet Joint-Motion architecture and checkpoint. Its ONNX exports include FP16 for WebGPU and FP32 for CPU fallback.
- FP16 conversion keeps model inputs and outputs in FP32 while converting floating-point weights and internal graph operations to FP16.

## Runtime

YOLO and ST-GCN++ run through ONNX Runtime Web. WebGPU is selected first and explicitly; when it cannot initialize, the matching FP32 model is loaded with the WASM CPU provider. The UI reports the provider actually selected by the YOLO inference session, not merely whether the browser exposes `navigator.gpu`.

## YOLO input/output contract

YOLO receives RGB float32 input shaped `[1,3,512,512]`, letterboxed from the source frame. The raw pose head returns 56 values per candidate: box center/size, person confidence, and 17 COCO keypoints. Browser-side decoding restores coordinates to the source frame, applies confidence filtering and NMS, and assigns lightweight track IDs.

## ST-GCN++ input/output contract

The action classifier receives `[1,1,2,100,17,3]`: one clip, two people, 100 frames, 17 COCO joints, and x/y/joint-motion confidence channels. It returns one of the original 60 NTU60 classes. The original checkpoint and two-person input contract are preserved.

## Verification

The deployment workflow exports both FP16 models, validates ONNX graphs, verifies the FP32 ST-GCN++ export against PyTorch, checks YOLO's fixed 512 input, and runs the browser smoke test. The smoke test runs in CI's Chromium environment; final GPU speed and WebGPU FP16 compatibility must also be measured on the target Android phone.

## Privacy

GitHub Pages serves static assets only. Model inference runs locally in dedicated workers. Camera frames and selected videos are not uploaded for inference.
