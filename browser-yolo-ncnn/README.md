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

- index.html, styles.css, app.js: browser UI and capability checks.
- tools/export_model.py: exports the official Nano pose model to NCNN FP16.
- model/: exported NCNN .param and .bin files will be placed here.
- runtime/: NCNN WebAssembly SIMD runtime and its JavaScript glue.
- wasm/: C++ inference bridge and Emscripten build files.

## Current status

This is the first project scaffold. The UI and model-export step are prepared. Browser inference is **not wired yet**: the NCNN WebAssembly SIMD runtime and the model-specific C++/JavaScript bridge must be built and tested before the page can run predictions.

## Export the model

On a machine with Python and the required build tools, run:

    python -m pip install --upgrade ultralytics
    python tools/export_model.py

The export script uses the official yolo26n-pose.pt weights and asks Ultralytics to export NCNN with FP16 precision. It then copies the generated .param, .bin, and metadata files into model/.

Official references:
- https://docs.ultralytics.com/tasks/pose/
- https://docs.ultralytics.com/integrations/ncnn/
- https://github.com/Tencent/ncnn/releases
- https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-webassembly

## Deployment

Publish this directory as a GitHub Pages site (or configure Pages to use this folder). The browser downloads the static runtime and model files, then runs inference locally. No video or image is uploaded to a server by this project.
