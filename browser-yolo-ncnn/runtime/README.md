# NCNN WebAssembly runtime

The deployment workflow builds two NCNN browser runtimes:

- SIMD + threads for cross-origin-isolated browsers with SharedArrayBuffer.
- Single-thread SIMD fallback for browsers without thread isolation.

These Emscripten builds use NCNN CPU execution. Vulkan is disabled in the WebAssembly build because the browser path does not expose NCNN's native Vulkan compute interface. The action-classification model uses a separate ONNX Runtime WebGPU execution provider when the browser supports WebGPU, with WASM CPU fallback.

Generated runtime files are placed here during deployment and are not committed to the source tree.
