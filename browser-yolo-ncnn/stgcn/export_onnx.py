from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    weights = MODEL_DIR / "stgcnpp_ntu60_xsub_hrnet.pth"
    output = MODEL_DIR / "stgcnpp_ntu60.onnx"
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
        wrapper,
        (sample,),
        str(output),
        input_names=["keypoint"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"keypoint": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    print(f"ONNX export: {output.stat().st_size:,} bytes")
    print("Input: [B,1,2,100,17,3] | Output: [B,60] | Opset: 17")

    if args.verify:
        session = ort.InferenceSession(
            str(output), providers=["CPUExecutionProvider"]
        )
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
            "verification": "passed",
            "batch_size": 2,
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "providers": session.get_providers(),
        }, indent=2))


if __name__ == "__main__":
    main()
