from __future__ import annotations

import json
from pathlib import Path
import shutil

import onnx
from onnx import TensorProto
from onnxconverter_common import float16
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
SOURCE_MODEL = "yolo26n-pose.pt"
IMAGE_SIZE = 512
EXPECTED_INPUT_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
EXPECTED_POSE_CHANNELS = 56


def _shape(value_info):
    return [
        dim.dim_value if dim.dim_value else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _assert_model_contract(graph: onnx.ModelProto, label: str) -> dict:
    onnx.checker.check_model(graph)
    if not graph.graph.input or not graph.graph.output:
        raise RuntimeError(f"{label} has no graph input/output.")

    input_info = graph.graph.input[0]
    output_info = graph.graph.output[0]
    input_shape = _shape(input_info)
    output_shape = _shape(output_info)

    if input_shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(
            f"{label} input must be {EXPECTED_INPUT_SHAPE}, got {input_shape}."
        )
    if EXPECTED_POSE_CHANNELS not in output_shape:
        raise RuntimeError(
            f"{label} pose output must contain {EXPECTED_POSE_CHANNELS} values "
            f"per candidate, got {output_shape}."
        )
    if input_info.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError(f"{label} input must remain FP32 for browser preprocessing.")
    if output_info.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise RuntimeError(f"{label} output must remain FP32 for browser decoding.")

    fp16_initializers = sum(
        tensor.data_type == TensorProto.FLOAT16 for tensor in graph.graph.initializer
    )
    if fp16_initializers == 0:
        raise RuntimeError(f"{label} contains no FP16 weights.")

    return {
        "input_name": input_info.name,
        "output_name": output_info.name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "fp16_initializers": fp16_initializers,
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(SOURCE_MODEL)

    # Export the raw pose head so the browser performs confidence/NMS decoding.
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

    fp32_graph = onnx.load(str(fp32_path))
    fp32_contract = _assert_model_contract(fp32_graph, "YOLO FP32 ONNX")

    fp16_graph = float16.convert_float_to_float16(fp32_graph, keep_io_types=True)
    fp16_contract = _assert_model_contract(fp16_graph, "YOLO FP16 ONNX")
    onnx.save(fp16_graph, str(fp16_path))

    report = {
        "model": SOURCE_MODEL,
        "input_size": IMAGE_SIZE,
        "input_shape": fp16_contract["input_shape"],
        "input_name": fp16_contract["input_name"],
        "output_name": fp16_contract["output_name"],
        "output_shape": fp16_contract["output_shape"],
        "precision": "FP16 weights/internal operators; FP32 input/output",
        "fp16_initializers": fp16_contract["fp16_initializers"],
        "fp32_bytes": fp32_path.stat().st_size,
        "fp16_bytes": fp16_path.stat().st_size,
        "fp32_contract": fp32_contract,
    }
    (MODEL_DIR / "yolo-onnx-config.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))

    if fp16_path.stat().st_size >= fp32_path.stat().st_size:
        raise RuntimeError("FP16 ONNX export did not reduce model size.")


if __name__ == "__main__":
    main()
