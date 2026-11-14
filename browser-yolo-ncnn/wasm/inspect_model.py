import json
from pathlib import Path

model_dir = Path(__file__).resolve().parents[1] / "model"
param = model_dir / "model.param"
lines = [line.strip() for line in param.read_text().splitlines()
         if line.strip() and not line.lstrip().startswith("#")]

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
    tops = parts[top_start:top_start + top_count]
    bottoms = parts[bottom_start:top_start]
    if len(tops) == top_count:
        layers.append((parts[0], bottoms, tops))

if not layers:
    raise RuntimeError("Could not parse NCNN model graph")

input_names = []
all_tops = set()
all_bottoms = set()
for layer_type, bottoms, tops in layers:
    all_tops.update(tops)
    all_bottoms.update(bottoms)
    if layer_type == "Input":
        input_names.extend(tops)

outputs = sorted(all_tops - all_bottoms)
if len(input_names) != 1:
    raise RuntimeError(f"Expected one model input, found {input_names}")
if len(outputs) != 1:
    raise RuntimeError(f"Expected one model output, found {outputs}")

config = {
    "input": input_names[0],
    "output": outputs[0],
    "input_size": 416,
    "keypoints": 17,
    "values_per_candidate": 56
}
(model_dir / "model-config.json").write_text(json.dumps(config, indent=2) + "\n")
print(json.dumps(config, indent=2))
