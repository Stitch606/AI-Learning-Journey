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
- runtime/: NCNN WebAssembly SIMD runtime and its JavaScript glue.
- wasm/: C++ inference bridge and Emscripten build files.

## Current status

The browser UI and streaming pose renderer are present. Browser inference is **not wired yet**: the repository currently has no NCNN WebAssembly runtime, no exported `.param`/`.bin` model pair, and no model-specific C++/JavaScript bridge. The renderer accepts detections through `window.publishPoseDetections(...)`, but nothing in the current page produces those detections yet. Do not treat camera preview or the renderer as model inference.

## Export the model

On a machine with Python and the required build tools, run:

    python -m pip install --upgrade ultralytics
    python tools/export_model.py

The export script uses the official yolo26n-pose.pt weights and asks Ultralytics to export NCNN with FP16 precision (`quantize=16`, the documented NCNN export setting). It then copies the generated .param, .bin, and metadata files into model/.

Official references:
- https://docs.ultralytics.com/tasks/pose/
- https://docs.ultralytics.com/integrations/ncnn/
- https://github.com/Tencent/ncnn/releases
- https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-webassembly

## Deployment

Publish this directory as a GitHub Pages site (or configure Pages to use this folder). The browser downloads the static runtime and model files, then runs inference locally. No video or image is uploaded to a server by this project.

## Streaming renderer contract

The inference bridge should publish each completed frame as:

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
