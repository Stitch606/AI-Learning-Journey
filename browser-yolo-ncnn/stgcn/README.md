# Browser ST-GCN++ NTU60

This folder preserves the original two-person PYSKL-compatible ST-GCN++ model
and its original NTU60 XSub HRNet Joint-Motion checkpoint.

- `model.py`: byte-for-byte copy of `GCN/STGCNPP/model.py`.
- `../model/stgcnpp_ntu60_xsub_hrnet.pth`: copied original checkpoint.
- `../model/stgcnpp_ntu60.onnx`: generated ONNX export for browser inference.

The input contract is fixed to `[B,1,2,100,17,3]`; the classifier returns
60 NTU60 logits. The export wrapper calls the original `forward_logits`
without changing the architecture, person pooling, weights, or preprocessing.

Run from the repository root:

```bash
python -m pip install torch numpy onnx onnxruntime
python browser-yolo-ncnn/stgcn/export_onnx.py --verify
```

The verification compares ONNX Runtime CPU outputs against the original
PyTorch model. Browser acceleration is selected separately by the browser
runtime; native NCNN Vulkan is not the browser WebAssembly backend.
