from pathlib import Path
import json
import shutil

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
SOURCE_MODEL = "yolo26n-pose.pt"
IMAGE_SIZE = 416


def read_model_io(param_path: Path) -> tuple[str, str]:
    lines = [
        line.strip()
        for line in param_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    layers = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            bottom_count = int(parts[2])
            top_count = int(parts[3])
        except ValueError:
            continue
        bottom_start = 4
        top_start = bottom_start + bottom_count
        bottoms = parts[bottom_start:top_start]
        tops = parts[top_start:top_start + top_count]
        if len(tops) == top_count:
            layers.append((parts[0], bottoms, tops))

    inputs = [name for layer_type, _, tops in layers
              if layer_type == "Input" for name in tops]
    all_tops = {name for _, _, tops in layers for name in tops}
    all_bottoms = {name for _, bottoms, _ in layers for name in bottoms}
    outputs = sorted(all_tops - all_bottoms)

    if len(inputs) != 1:
        raise RuntimeError(f"Expected one NCNN input blob, found {inputs}")
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one NCNN output blob, found {outputs}")
    return inputs[0], outputs[0]


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(SOURCE_MODEL)
    exported = Path(model.export(
        format="ncnn",
        quantize=16,
        imgsz=IMAGE_SIZE,
        device="cpu",
    ))

    param_files = list(exported.glob("*.param"))
    bin_files = list(exported.glob("*.bin"))
    if len(param_files) != 1 or len(bin_files) != 1:
        raise RuntimeError(
            f"Expected exactly one NCNN param/bin pair in {exported}; "
            f"found {len(param_files)} param and {len(bin_files)} bin files"
        )

    param_path = MODEL_DIR / "model.param"
    bin_path = MODEL_DIR / "model.bin"
    shutil.copy2(param_files[0], param_path)
    shutil.copy2(bin_files[0], bin_path)

    input_blob, output_blob = read_model_io(param_path)
    config = {
        "input": input_blob,
        "output": output_blob,
        "input_size": IMAGE_SIZE,
        "keypoints": 17,
        "values_per_candidate": 56,
        "precision": "FP16",
        "model": SOURCE_MODEL,
    }
    (MODEL_DIR / "model-config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    print(json.dumps(config, indent=2))
    for path in (param_path, bin_path, MODEL_DIR / "model-config.json"):
        print(f"{path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
