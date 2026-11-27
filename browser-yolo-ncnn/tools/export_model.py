from __future__ import annotations

import json
from pathlib import Path
import shutil

import onnx
from onnxconverter_common import float16
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
SOURCE_MODEL = "yolo26n-pose.pt"
IMAGE_SIZE = 512


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(SOURCE_MODEL)

    # Export the raw pose head so the browser performs the same confidence/NMS
    # decoding as the previous NCNN bridge.
    head = model.model.model[-1]
    if hasattr(head, "end2end"):
        head.end2end = False

    exported = Path(model.export(
        format="onnx",
        imgsz=IMAGE_SIZE,
        half=False,
        dynamic=False,
        simplify=True,
        opset=17,
        device="cpu",
    ))
    fp32_path = MODEL_DIR / "yolo26n-pose-512-fp32.onnx"
    fp16_path = MODEL_DIR / "yolo26n-pose-512-fp16.onnx"
    shutil.copy2(exported, fp32_path)

    graph = onnx.load(str(fp32_path))
    onnx.checker.check_model(graph)
    converted = float16.convert_float_to_float16(graph, keep_io_types=True)
    onnx.checker.check_model(converted)
    onnx.save(converted, str(fp16_path))

    input_info = graph.graph.input[0]
    output_info = graph.graph.output[0]
    report = {
        "model": SOURCE_MODEL,
        "input_size": IMAGE_SIZE,
        "input_name": input_info.name,
        "output_name": output_info.name,
        "precision": "FP16 weights and internal floating-point operators; FP32 input/output",
        "fp32_bytes": fp32_path.stat().st_size,
        "fp16_bytes": fp16_path.stat().st_size,
        "output_shape": [d.dim_value or d.dim_param for d in output_info.type.tensor_type.shape.dim],
    }
    (MODEL_DIR / "yolo-onnx-config.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if fp16_path.stat().st_size >= fp32_path.stat().st_size:
        raise RuntimeError("FP16 ONNX export did not reduce model size.")


if __name__ == "__main__":
    main()
