from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto
import onnxruntime as ort
import torch
from onnxconverter_common import float16
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))
from model import STGCNPP


class ExportWrapper(nn.Module):
    def __init__(self, model: STGCNPP):
        super().__init__()
        self.model = model

    def forward(self, keypoint):
        return self.model.forward_logits(keypoint)


def validate_fp16_graph(graph: onnx.ModelProto, label: str) -> int:
    onnx.checker.check_model(graph)
    fp16_initializers = sum(
        tensor.data_type == TensorProto.FLOAT16 for tensor in graph.graph.initializer
    )
    if fp16_initializers == 0:
        raise RuntimeError(f"{label} contains no FP16 weights.")
    input_info = graph.graph.input[0]
    output_info = graph.graph.output[0]
    if input_info.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError(f"{label} input must remain FP32.")
    if output_info.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError(f"{label} output must remain FP32.")
    return fp16_initializers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    weights = MODEL_DIR / "stgcnpp_ntu60_xsub_hrnet.pth"
    fp32_output = MODEL_DIR / "stgcnpp_ntu60-fp32.onnx"
    fp16_output = MODEL_DIR / "stgcnpp_ntu60-fp16.onnx"
    if not weights.is_file():
        raise FileNotFoundError(weights)

    torch.manual_seed(606)
    np.random.seed(606)
    model = STGCNPP(checkpoint_path=weights, num_classes=60, device="cpu", strict=True)
    model.eval()
    wrapper = ExportWrapper(model).eval()
    sample = torch.zeros((1, 1, 2, 100, 17, 3), dtype=torch.float32)
    sample[..., 2] = 0.5

    torch.onnx.export(
        wrapper, (sample,), str(fp32_output),
        input_names=["keypoint"], output_names=["logits"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes={"keypoint": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    graph = onnx.load(str(fp32_output))
    onnx.checker.check_model(graph)
    converted = float16.convert_float_to_float16(graph, keep_io_types=True)
    fp16_initializers = validate_fp16_graph(converted, "ST-GCN++ FP16 ONNX")
    onnx.save(converted, str(fp16_output))

    print(json.dumps({
        "fp32_bytes": fp32_output.stat().st_size,
        "fp16_bytes": fp16_output.stat().st_size,
        "fp16_initializers": fp16_initializers,
        "input": "[B,1,2,100,17,3]",
        "output": "[B,60]",
        "precision": "FP16 weights/internal operators; FP32 input/output",
    }, indent=2))
    if fp16_output.stat().st_size >= fp32_output.stat().st_size:
        raise RuntimeError("ST-GCN++ FP16 ONNX export did not reduce model size.")

    if args.verify:
        session = ort.InferenceSession(str(fp32_output), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(606)
        test_input = rng.random((2, 1, 2, 100, 17, 3), dtype=np.float32)
        with torch.inference_mode():
            expected = wrapper(torch.from_numpy(test_input)).cpu().numpy()
        actual = session.run(["logits"], {"keypoint": test_input})[0]
        max_abs = float(np.max(np.abs(expected - actual)))
        mean_abs = float(np.mean(np.abs(expected - actual)))
        if not np.allclose(expected, actual, rtol=2e-3, atol=2e-4):
            raise AssertionError(
                f"ONNX/PyTorch mismatch: max_abs={max_abs:.8g}, mean_abs={mean_abs:.8g}"
            )
        print(json.dumps({
            "fp32_verification": "passed",
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
        }, indent=2))


if __name__ == "__main__":
    main()
