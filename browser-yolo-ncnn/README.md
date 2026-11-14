# Browser YOLO Pose — NCNN + WebAssembly SIMD

A browser-only YOLO pose demo. Inference is intended to run on the visitor's device; GitHub Pages is static hosting only.

## Fixed project choices

- Model: official Ultralytics **YOLO26n-pose** (Nano pose)
- Inference engine: **Tencent NCNN**
- Browser runtime: **WebAssembly SIMD** (single-threaded build first)
- Model precision: **FP16** (quantize=16)
- Initial input size: **416 × 416** as a mobile-oriented starting point; benchmark against 640 before deciding the final default.
- No inference server and no ONNX Runtime.

## Project layout

- index.html, styles.css, app.js: browser UI, camera/image input, and streaming inference hand-off.
- renderer.js: real-time COCO-17 skeleton, per-person boxes/labels, confidence filtering, high-DPI scaling, and optional track-ID smoothing.
- tools/export_model.py: exports the official Nano pose model to NCNN FP16.
- model/: exported NCNN .param and .bin files will be placed here.
- inference-worker.js: off-main-thread model loading, frame preprocessing, inference and coordinate restoration.
- runtime/: generated NCNN WebAssembly SIMD module and `.wasm` binary (created by Actions).
- wasm/pose_bridge.cpp: NCNN C++ inference bridge and post-processing.
- wasm/inspect_model.py: discovers the actual NCNN input/output blob names.

## Current status

The browser app now has a real inference pipeline: GitHub Actions exports YOLO26n-Pose to NCNN FP16, builds NCNN as single-threaded WebAssembly SIMD, compiles the C++ bridge, and publishes the generated model/runtime with GitHub Pages. The browser loads the model locally in a Web Worker, preprocesses camera/image frames with letterboxing, runs NCNN, decodes person boxes and 17 COCO keypoints, applies confidence filtering and NMS, and sends per-person detections to the streaming canvas renderer. Runtime and model binaries are generated during deployment and are not committed to the source tree.

## Export the model

The deployment workflow performs the export automatically. It uses the official `yolo26n-pose.pt` weights and exports NCNN at 416 × 416 with FP16 precision (`quantize=16`). It also inspects the generated NCNN graph to discover its input and output blob names. NCNN export disables the YOLO26 end-to-end branch for this format, so the C++ bridge handles the raw one-to-many pose tensor and performs confidence filtering and NMS itself.

Official references:
- https://docs.ultralytics.com/tasks/pose/
- https://docs.ultralytics.com/integrations/ncnn/
- https://github.com/Tencent/ncnn/releases
- https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-webassembly

## Deployment

Publish this directory as a GitHub Pages site (or configure Pages to use this folder). The browser downloads the static runtime and model files, then runs inference locally. No video or image is uploaded to a server by this project.

## Streaming renderer contract

The browser worker publishes each completed frame to the renderer as:

```js
window.publishPoseDetections({
  sourceWidth: 416,
  sourceHeight: 416,
  detections: [{
    trackId: 0,
    label: "Person",
    score: 0.94,
    box: [x1, y1, x2, y2],
    keypoints: [[x, y, confidence], ...17 COCO points]
  }]
});
```

All coordinates must use the same source-image coordinate system. The renderer draws each person independently and does not average detections. Stable `trackId` values enable temporal smoothing; without them, points are drawn directly to avoid mixing identities when detection order changes.

## Build and deployment

The Pages workflow builds NCNN from Tencent's source with `NCNN_THREADS=OFF` and the WebAssembly SIMD extension enabled, then compiles `wasm/pose_bridge.cpp` with Emscripten. The runtime and FP16 model are generated into the Pages artifact during the workflow. Camera frames are transferred to a dedicated Web Worker; if inference is still running, the next frame is skipped rather than queued, prioritizing low latency.
