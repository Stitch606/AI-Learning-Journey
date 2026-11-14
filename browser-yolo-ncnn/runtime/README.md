# NCNN WebAssembly runtime

This folder is reserved for the official Tencent NCNN **WebAssembly SIMD** runtime build and its JavaScript loader/glue files.

The official NCNN release archive provides WebAssembly builds with SIMD variants. Use the SIMD-only build for this GitHub Pages project; do not include the threads build.

Runtime files are intentionally not fabricated or replaced with an unrelated inference engine. The exact official archive layout and exported module interface must be inspected before wiring the browser bridge.

Source: https://github.com/Tencent/ncnn/releases
