from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from GCN.STGCNPP.model_multi_person import STGCNPPMultiPerson

KEYPOINTS_PATH = ROOT / "yolo-artifact" / "person_keypoints.npz"
CHECKPOINT = ROOT / "GCN" / "checkpoints" / "stgcnpp_jm_ntu60_xsub_hrnet_finetuned.pth"
OUTPUT_DIR = ROOT / "artifacts" / "kth_deterministic_window_trace"
REPORT_PATH = OUTPUT_DIR / "report.json"
WINDOW_SIZE = 100
MAX_PERSONS = 10
IMAGE_SIZE = (160, 120)

WINDOWS = [
    (0, 100),
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 500),
]

def tensor_stats(value):
    if not torch.is_tensor(value):
        return {"type": type(value).__name__}
    x = value.detach()
    if x.numel() == 0:
        return {"shape": list(x.shape), "dtype": str(x.dtype), "numel": 0}
    y = x.float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "numel": int(x.numel()),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "mean": float(y.mean().item()),
        "std": float(y.std(unbiased=False).item()),
        "nan_count": int(torch.isnan(y).sum().item()),
        "inf_count": int(torch.isinf(y).sum().item()),
        "zero_count": int((y == 0).sum().item()),
    }

def numpy_stats(value):
    x = np.asarray(value)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "zero_count": int((x == 0).sum()),
        "mean_joint_confidence": float(np.mean(x[..., 2])),
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    if not KEYPOINTS_PATH.is_file():
        raise FileNotFoundError(f"YOLO keypoint artifact missing: {KEYPOINTS_PATH}")
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Fine-tuned checkpoint missing: {CHECKPOINT}")

    checkpoint_data = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    checkpoint_metadata = {
        "epoch": checkpoint_data.get("epoch") if isinstance(checkpoint_data, dict) else None,
        "best_validation_accuracy": checkpoint_data.get("best_validation_accuracy") if isinstance(checkpoint_data, dict) else None,
        "fine_tuned": checkpoint_data.get("fine_tuned") if isinstance(checkpoint_data, dict) else None,
    }

    model = STGCNPPMultiPerson(
        checkpoint_path=str(CHECKPOINT),
        device="cpu",
        max_persons=MAX_PERSONS,
        strict=True,
    ).eval()

    hook_events = []
    def make_pre_hook(name):
        def hook(module, inputs):
            tensors = [tensor_stats(item) for item in inputs if torch.is_tensor(item)]
            hook_events.append({
                "stage": name,
                "event": "input",
                "tensors": tensors,
            })
        return hook

    def make_post_hook(name):
        def hook(module, inputs, output):
            values = output if isinstance(output, (tuple, list)) else (output,)
            tensors = [tensor_stats(item) for item in values if torch.is_tensor(item)]
            hook_events.append({
                "stage": name,
                "event": "output",
                "tensors": tensors,
            })
        return hook

    handles = []
    for name, module in [
        ("backbone", model.backbone),
        ("cls_head", model.cls_head),
        ("cls_head.fc_cls", model.cls_head.fc_cls),
    ]:
        handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
        handles.append(module.register_forward_hook(make_post_hook(name)))

    all_predictions = []
    track_summaries = []
    profiler_summaries = []
    start_all = time.perf_counter()

    try:
        archive = np.load(KEYPOINTS_PATH, allow_pickle=False)
        track_keys = sorted(archive.files)
        if not track_keys:
            raise RuntimeError("YOLO artifact contains no person tracks")

        for track_key in track_keys:
            sequence = np.asarray(archive[track_key], dtype=np.float32)
            if sequence.ndim != 3 or sequence.shape[1:] != (17, 3):
                raise ValueError(f"{track_key}: expected [T,17,3], got {sequence.shape}")
            if not np.isfinite(sequence).all():
                raise ValueError(f"{track_key}: non-finite YOLO keypoints")
            if np.any(sequence[..., 2] < 0) or np.any(sequence[..., 2] > 1):
                raise ValueError(f"{track_key}: confidence outside [0,1]")

            total_frames = len(sequence)
            if total_frames < WINDOW_SIZE:
                track_summaries.append({
                    "track": track_key,
                    "frames": total_frames,
                    "status": "skipped_less_than_100_frames",
                })
                continue

            starts = [start for start, _ in WINDOWS if start + WINDOW_SIZE <= total_frames]
            tail_start = total_frames - WINDOW_SIZE
            if tail_start not in starts:
                starts.append(tail_start)

            track_results = []
            for window_index, start in enumerate(starts, start=1):
                end = start + WINDOW_SIZE
                window = sequence[start:end].copy()
                if len(window) != WINDOW_SIZE:
                    raise AssertionError(f"Window {window_index} has {len(window)} frames, not 100")

                # The exact 100-frame input bypasses temporal subsampling in _prepare_single_person.
                prepared = model._prepare_single_person(window, image_size=IMAGE_SIZE)
                if prepared.shape != (100, 17, 3):
                    raise AssertionError(f"Prepared JM shape mismatch: {prepared.shape}")

                event_start = len(hook_events)
                profiler_path = OUTPUT_DIR / f"{track_key}_window_{window_index}_trace.json"
                started = time.perf_counter()
                with profile(
                    activities=[ProfilerActivity.CPU],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                ) as prof:
                    result = model.predict(
                        window,
                        topk=5,
                        person_ids=[int(track_key.split("_")[-1])],
                        image_size=IMAGE_SIZE,
                    )
                elapsed = time.perf_counter() - started
                prof.export_chrome_trace(str(profiler_path))

                person = result["persons"][0]
                prediction = {
                    "track": track_key,
                    "window_index": window_index,
                    "start_frame_inclusive": start,
                    "end_frame_exclusive": end,
                    "frame_numbers_inclusive": [start, end - 1],
                    "input_raw_keypoints": numpy_stats(window),
                    "prepared_joint_motion": numpy_stats(prepared),
                    "top1": person["action"],
                    "top1_class_id": int(person["class_id"]),
                    "top1_confidence": float(person["confidence"]),
                    "matches_ground_truth": person["action"].strip().lower() == "hand waving",
                    "top5": [
                        {
                            "action": item["action"],
                            "class_id": int(item["class_id"]),
                            "confidence": float(item["confidence"]),
                        }
                        for item in person["top_predictions"]
                    ],
                    "inference_ms": round(elapsed * 1000, 4),
                    "forward_hook_events": hook_events[event_start:],
                    "profiler_trace": str(profiler_path.relative_to(ROOT)),
                    "profiler_top_ops": [
                        {
                            "operator": item.key,
                            "cpu_time_total_us": float(item.cpu_time_total),
                            "calls": int(item.count),
                        }
                        for item in prof.key_averages()[:12]
                    ],
                }
                all_predictions.append(prediction)
                track_results.append(prediction)
                print(
                    f"{track_key} window {window_index}/{len(starts)} "
                    f"frames={start}-{end-1}: {person['action']} "
                    f"{person['confidence']:.6f}",
                    flush=True,
                )

            track_summaries.append({
                "track": track_key,
                "frames": total_frames,
                "windows_processed": len(track_results),
                "windows": [
                    {
                        "window_index": item["window_index"],
                        "frames": item["frame_numbers_inclusive"],
                        "top1": item["top1"],
                        "confidence": item["top1_confidence"],
                        "matches_ground_truth": item["matches_ground_truth"],
                    }
                    for item in track_results
                ],
            })
    finally:
        for handle in handles:
            handle.remove()

    successful = all_predictions
    report = {
        "test": "KTH Hand Waving: deterministic contiguous 100-frame GCN windows",
        "ground_truth": "Hand Waving",
        "sampling_policy": {
            "random_sampling": False,
            "window_size": WINDOW_SIZE,
            "windows": "contiguous fixed 100-frame slices; final window is anchored to the final frame and may overlap the preceding window",
            "starts_for_581_frames": [0, 100, 200, 300, 400, 481],
            "all_frames_covered": True,
            "last_window_includes_final_frame": True,
            "overlap_frames": 19,
        },
        "source": {
            "keypoint_file": str(KEYPOINTS_PATH.relative_to(ROOT)),
            "tracks": track_keys,
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_bytes": CHECKPOINT.stat().st_size,
            "checkpoint_metadata": checkpoint_metadata,
            "image_size_width_height": list(IMAGE_SIZE),
        },
        "instrumentation": {
            "forward_hooks": ["backbone", "cls_head", "cls_head.fc_cls"],
            "tensor_stats": ["shape", "dtype", "min", "max", "mean", "std", "NaN", "Inf", "zero count"],
            "profiler": "torch.profiler CPU, record_shapes=True, profile_memory=True; Chrome trace per window",
            "hook_event_count": len(hook_events),
        },
        "tracks": track_summaries,
        "window_predictions": all_predictions,
        "summary": {
            "windows_evaluated": len(successful),
            "windows_matching_ground_truth": sum(item["matches_ground_truth"] for item in successful),
            "windows_not_matching_ground_truth": sum(not item["matches_ground_truth"] for item in successful),
            "mean_inference_ms": float(np.mean([item["inference_ms"] for item in successful])) if successful else None,
            "total_wall_seconds": round(time.perf_counter() - start_all, 4),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    main()
