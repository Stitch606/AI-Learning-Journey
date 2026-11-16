from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "huggingface-space/process/models/ST_GCN++.py"
CHECKPOINT = ROOT / "huggingface-space/process/weights/stgcnpp_jm_ntu60_xsub_hrnet_finetuned.pth"
OUT = ROOT / "artifacts/ncnn_gcn_preflight"
REPORT = OUT / "gcn_ncnn_preflight.json"


def load_model():
    spec = importlib.util.spec_from_file_location("project_stgcnpp_preflight", ARCH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ST-GCN++ architecture: {ARCH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.STGCNPPMultiPerson(
        checkpoint_path=str(CHECKPOINT), device="cpu", max_persons=10
    )
    model.eval()
    return model


class OnePersonExport(torch.nn.Module):
    """Export only the tensor model; raw-pose preprocessing remains outside NCNN."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: [100,17,3], already normalized and converted to Joint Motion.
        x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        logits = self.model._person_clip_logits(x)  # [1,1,1,60]
        return torch.softmax(logits.mean(dim=1).squeeze(0).squeeze(0), dim=-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < 1024:
        raise RuntimeError(f"Fine-tuned checkpoint missing: {CHECKPOINT}")
    if not shutil_which("pnnx"):
        raise RuntimeError("pnnx executable is not installed.")
    try:
        import ncnn
    except Exception as exc:
        raise RuntimeError(f"Python NCNN runtime import failed: {exc}") from exc

    torch.set_num_threads(2)
    model = load_model()
    export_model = OnePersonExport(model).eval()
    rng = np.random.default_rng(606)
    sample = rng.normal(0.0, 0.15, size=(100, 17, 3)).astype(np.float32)
    sample[..., 2] = rng.uniform(0.05, 1.0, size=(100, 17)).astype(np.float32)
    tensor = torch.from_numpy(sample)

    with torch.inference_mode():
        expected = export_model(tensor).cpu().numpy()

    traced_path = OUT / "stgcnpp_one_person.pt"
    traced = torch.jit.trace(export_model, tensor, strict=False)
    traced.save(str(traced_path))

    started = time.perf_counter()
    command = [
        "pnnx", str(traced_path),
        "inputshape=[100,17,3]",
        "device=cpu",
    ]
    completed = subprocess.run(
        command, cwd=OUT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False
    )
    (OUT / "pnnx_conversion.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            "PNNX could not convert ST-GCN++ to NCNN. See pnnx_conversion.log.\n"
            + completed.stdout[-12000:]
        )

    param_path = OUT / "stgcnpp_one_person.ncnn.param"
    bin_path = OUT / "stgcnpp_one_person.ncnn.bin"
    if not param_path.is_file() or not bin_path.is_file():
        candidates = sorted(str(p.name) for p in OUT.glob("*.ncnn.*"))
        raise RuntimeError(f"PNNX did not produce expected NCNN files: {candidates}")

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 2
    if net.load_param(str(param_path)) != 0 or net.load_model(str(bin_path)) != 0:
        raise RuntimeError("NCNN failed to load the converted ST-GCN++ model.")

    extractor = net.create_extractor()
    input_mat = ncnn.Mat(np.ascontiguousarray(sample))
    ret = extractor.input("in0", input_mat)
    if ret != 0:
        raise RuntimeError(f"NCNN input failed with code {ret}.")
    ret, output_mat = extractor.extract("out0")
    if ret != 0:
        raise RuntimeError(f"NCNN output extraction failed with code {ret}.")
    actual = np.asarray(output_mat).reshape(-1).astype(np.float32)
    if actual.shape != expected.reshape(-1).shape:
        raise RuntimeError(
            f"Output shape mismatch: PyTorch={expected.shape}, NCNN={actual.shape}"
        )

    absolute = np.abs(actual - expected.reshape(-1))
    max_abs = float(absolute.max())
    mean_abs = float(absolute.mean())
    report = {
        "status": "passed" if np.allclose(actual, expected.reshape(-1), atol=2e-3, rtol=2e-3) else "parity_failed",
        "architecture": str(ARCH.relative_to(ROOT)),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "input_shape": list(sample.shape),
        "output_shape": list(actual.shape),
        "max_absolute_error": max_abs,
        "mean_absolute_error": mean_abs,
        "tolerance": {"atol": 2e-3, "rtol": 2e-3},
        "conversion_seconds": round(time.perf_counter() - started, 4),
        "runtime": "NCNN CPU, 2 threads",
        "note": "This is a graph-conversion gate using a deterministic prepared Joint-Motion tensor. It does not yet measure the complete video pipeline.",
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("NCNN/PyTorch ST-GCN++ parity check failed.")


def shutil_which(name):
    import shutil
    return shutil.which(name)


if __name__ == "__main__":
    main()
