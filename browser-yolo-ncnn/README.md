# Browser YOLO26 Pose + ST-GCN++ Action Recognition

A browser-local vision pipeline hosted as static files on GitHub Pages. Camera frames remain on the visitor's device.

## Models

- **YOLO26n-Pose**: exported to NCNN FP16 at 416x416. The browser runs the NCNN build as WebAssembly SIMD, using multiple CPU threads when cross-origin isolation is available and a single-thread fallback otherwise.
- **ST-GCN++**: the original two-person NTU60 XSub HRNet Joint-Motion architecture and checkpoint are preserved. The model is exported to ONNX without changing its weights or two-person input contract. ONNX Runtime Web uses WebGPU when available and falls back to WebAssembly CPU.

## ST-GCN++ input and output

The action classifier receives `[1,1,2,100,17,3]`: one clip, two people, 100 frames, 17 COCO joints, and x/y/joint-motion confidence channels. It returns one of the original 60 NTU60 scene-level classes. The UI draws a group rectangle around the two selected people; it does not claim to identify which person performed the action.

The source architecture is `stgcn/model.py`, and the original checkpoint is copied to `model/stgcnpp_ntu60_xsub_hrnet.pth`. The deployment workflow exports `model/stgcnpp_ntu60.onnx` and runs the browser smoke test.

## Browser acceleration

Browsers do not expose raw Vulkan or OpenGL compute APIs to web applications. NCNN's browser WebAssembly build therefore uses CPU SIMD/threads. GPU inference in the browser uses WebGPU through ONNX Runtime Web; WebGPU is the browser API that can map to native Vulkan, Metal, or Direct3D backends depending on the device and browser.

## Privacy and deployment

GitHub Pages serves static assets only. Model inference runs locally in dedicated workers. No camera frame or selected image is uploaded for inference.
