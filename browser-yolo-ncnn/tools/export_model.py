from pathlib import Path
import shutil

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
SOURCE_MODEL = "yolo26n-pose.pt"
IMAGE_SIZE = 416


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(SOURCE_MODEL)
    exported = Path(
        model.export(
            format="ncnn",
            quantize=16,
            imgsz=IMAGE_SIZE,
            device="cpu",
        )
    )

    files = list(exported.glob("*.param")) + list(exported.glob("*.bin"))
    files += list(exported.glob("*.yaml"))
    if not any(path.suffix == ".param" for path in files):
        raise RuntimeError("No NCNN .param file found in " + str(exported))
    if not any(path.suffix == ".bin" for path in files):
        raise RuntimeError("No NCNN .bin file found in " + str(exported))

    for source in files:
        shutil.copy2(source, MODEL_DIR / source.name)

    print("NCNN FP16 model exported to: " + str(MODEL_DIR))
    for path in sorted(MODEL_DIR.iterdir()):
        if path.is_file():
            print(path.name + ": " + format(path.stat().st_size, ",") + " bytes")


if __name__ == "__main__":
    main()
