from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from GCN.STGCNPP.model_multi_person import STGCNPPMultiPerson

KEYPOINTS_PATH = ROOT / "yolo-artifact" / "person_keypoints.npz"
CHECKPOINT = ROOT / "GCN" / "checkpoints" / "stgcnpp_jm_ntu60_xsub_hrnet_finetuned.pth"
OUTPUT_DIR = ROOT / "artifacts" / "kth_pyskl_full_sequence_jm"
REPORT_PATH = OUTPUT_DIR / "report.json"
WINDOW_SIZE = 100
IMAGE_SIZE = (160, 120)
WINDOW_STARTS = [0, 100, 200, 300, 400, 481]


def pyskl_reference_joint_motion(sequence, image_size):
    """Independent equivalent of PYSKL PreNormalize2D + GenSkeFeat(jm)."""
    raw = np.asarray(sequence, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (17, 3):
        raise ValueError(f"Expected [T,17,3], got {raw.shape}")

    keypoint = raw[None, ..., :2].copy()
    score = raw[None, ..., 2].copy()
    keypoint = np.concatenate([keypoint, score[..., None]], axis=-1)

    maskout = keypoint[..., 2] <= 0.01
    h, w = image_size[1], image_size[0]
    keypoint[..., 0] = (keypoint[..., 0] - (w / 2.0)) / (w / 2.0)
    keypoint[..., 1] = (keypoint[..., 1] - (h / 2.0)) / (h / 2.0)
    keypoint[..., 0][maskout] = 0.0
    keypoint[..., 1][maskout] = 0.0

    motion = np.zeros_like(keypoint, dtype=np.float32)
    if raw.shape[0] > 1:
        motion[:, :-1] = np.diff(keypoint, axis=1)
        motion[:, :-1, :, 2] = (
            keypoint[:, :-1, :, 2] + keypoint[:, 1:, :, 2]
        ) / 2.0
    return motion[0]


def array_stats(value):
    x = np.asarray(value, dtype=np.float32)
    return {
        "shape": list(x.shape),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "channel_stats": {
            name: {
                "min": float(x[..., i].min()),
                "max": float(x[..., i].max()),
                "mean": float(x[..., i].mean()),
                "std": float(x[..., i].std()),
            }
            for i, name in enumerate(("joint_motion_x", "joint_motion_y", "mean_joint_confidence"))
        },
    }


def tensor_stats(value):
    if not torch.is_tensor(value):
        return {"type": type(value).__name__}
    x = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "nan_count": int(torch.isnan(x).sum()),
        "inf_count": int(torch.isinf(x).sum()),
        "zero_count": int((x == 0).sum()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    archive = np.load(KEYPOINTS_PATH, allow_pickle=False)
    track_keys = sorted(archive.files)
    if len(track_keys) != 1:
        raise RuntimeError(f"Expected the single KTH person track, got {track_keys}")

    model = STGCNPPMultiPerson(
        checkpoint_path=str(CHECKPOINT),
        device="cpu",
        max_persons=10,
        strict=True,
    ).eval()

    all_predictions = []
    track_reports = []
    hook_events = []
    handles = []

    def make_pre_hook(name):
        def hook(module, inputs):
            hook_events.append({
                "stage": name,
                "event": "input",
                "tensors": [tensor_stats(item) for item in inputs if torch.is_tensor(item)],
            })
        return hook

    def make_post_hook(name):
        def hook(module, inputs, output):
            values = output if isinstance(output, (tuple, list)) else (output,)
            hook_events.append({
                "stage": name,
                "event": "output",
                "tensors": [tensor_stats(item) for item in values if torch.is_tensor(item)],
            })
        return hook

    for name, module in [
        ("backbone", model.backbone),
        ("cls_head", model.cls_head),
        ("cls_head.fc_cls", model.cls_head.fc_cls),
    ]:
        handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
        handles.append(module.register_forward_hook(make_post_hook(name)))

    started_all = time.perf_counter()
    try:
        for track_key in track_keys:
            raw = np.asarray(archive[track_key], dtype=np.float32)
            if raw.shape != (581, 17, 3):
                raise ValueError(f"Expected KTH track [581,17,3], got {raw.shape}")

            # Compute JM once across all 581 frames, exactly as PYSKL does
            # before UniformSampleFrames. Do not restart differences at windows.
            model_full_jm = model._prepare_joint_motion_sequence(raw, IMAGE_SIZE)
            reference_full_jm = pyskl_reference_joint_motion(raw, IMAGE_SIZE)
            difference = np.abs(model_full_jm - reference_full_jm)
            parity = {
                "reference": "PYSKL PreNormalize2D(img_shape=(120,160)) -> GenSkeFeat(feats=['jm'])",
                "max_absolute_difference": float(difference.max()),
                "mean_absolute_difference": float(difference.mean()),
                "allclose_atol_1e-7": bool(np.allclose(
                    model_full_jm, reference_full_jm, rtol=0.0, atol=1e-7
                )),
                "model_full_sequence": array_stats(model_full_jm),
                "pyskl_reference_full_sequence": array_stats(reference_full_jm),
                "absolute_difference_by_channel": {
                    name: float(difference[..., i].max())
                    for i, name in enumerate(("joint_motion_x", "joint_motion_y", "mean_joint_confidence"))
                },
            }
            if not parity["allclose_atol_1e-7"]:
                raise AssertionError(
                    "Standalone preprocessing differs from PYSKL reference: "
                    + json.dumps(parity, indent=2)
                )

            results = []
            for window_index, start in enumerate(WINDOW_STARTS, start=1):
                end = start + WINDOW_SIZE
                if end > len(raw):
                    raise ValueError(f"Window {start}:{end} exceeds {len(raw)} source frames")

                # Slice the already-computed full-video JM tensor.
                window_jm = model_full_jm[start:end].copy()
                reference_window = reference_full_jm[start:end]
                if not np.array_equal(window_jm, reference_window):
                    raise AssertionError(f"Window {window_index} JM differs from PYSKL reference")

                event_start = len(hook_events)
                trace_path = OUTPUT_DIR / f"{track_key}_window_{window_index}_trace.json"
                window_started = time.perf_counter()
                with profile(
                    activities=[ProfilerActivity.CPU],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                ) as prof:
                    prediction_result = model.predict(
                        window_jm,
                        topk=5,
                        person_ids=[int(track_key.split("_")[-1])],
                        image_size=IMAGE_SIZE,
                        input_is_joint_motion=True,
                    )
                elapsed = time.perf_counter() - window_started
                prof.export_chrome_trace(str(trace_path))

                person = prediction_result["persons"][0]
                hand_waving_index = next(
                    index for index, name in enumerate(model.class_names)
                    if name.strip().lower() == "hand waving"
                )
                item = {
                    "window_index": window_index,
                    "frame_range_inclusive": [start, end - 1],
                    "input_type": "PYSKL Joint Motion already computed on all 581 frames",
                    "input_stats": array_stats(window_jm),
                    "top1": person["action"],
                    "top1_class_id": int(person["class_id"]),
                    "top1_confidence": float(person["confidence"]),
                    "matches_ground_truth": person["action"].strip().lower() == "hand waving",
                    "hand_waving_probability": float(
                        person["probabilities"][hand_waving_index]
                    ),
                    "top5": [
                        {
                            "action": row["action"],
                            "class_id": int(row["class_id"]),
                            "confidence": float(row["confidence"]),
                        }
                        for row in person["top_predictions"]
                    ],
                    "inference_ms": round(elapsed * 1000, 4),
                    "forward_hook_events": hook_events[event_start:],
                    "profiler_trace": str(trace_path.relative_to(ROOT)),
                    "profiler_top_ops": [
                        {
                            "operator": row.key,
                            "cpu_time_total_us": float(row.cpu_time_total),
                            "calls": int(row.count),
                        }
                        for row in prof.key_averages()[:12]
                    ],
                }
                all_predictions.append(item)
                results.append(item)
                print(
                    f"{track_key} window {window_index}/{len(WINDOW_STARTS)} "
                    f"frames={start}-{end-1}: {person['action']} "
                    f"{person['confidence']:.6f}",
                    flush=True,
                )

            track_reports.append({
                "track": track_key,
                "source_frames": len(raw),
                "pyskl_preprocessing_parity": parity,
                "windows": results,
            })
    finally:
        for handle in handles:
            handle.remove()

    # Diagnostic matrix: distinguish the fine-tuned weights from coordinate
    # normalization scale. The default PYSKL image shape is also tested explicitly.
    diagnostic_matrix = {}
    base_checkpoint = ROOT / "GCN" / "checkpoints" / "stgcnpp_jm_ntu60_xsub_hrnet.pth"
    default_shape_jm = pyskl_reference_joint_motion(raw, (1920, 1080))

    diagnostic_models = {
        "fine_tuned_actual_frame_shape": (model, model_full_jm),
        "fine_tuned_pyskl_default_shape_1920x1080": (model, default_shape_jm),
    }
    base_model = STGCNPPMultiPerson(
        checkpoint_path=str(base_checkpoint),
        device="cpu",
        max_persons=10,
        strict=True,
    ).eval()
    diagnostic_models["original_pretrained_actual_frame_shape"] = (base_model, model_full_jm)
    diagnostic_models["original_pretrained_pyskl_default_shape_1920x1080"] = (base_model, default_shape_jm)

    for variant_name, (variant_model, full_jm) in diagnostic_models.items():
        variant_rows = []
        for window_index, start in enumerate(WINDOW_STARTS, start=1):
            end = start + WINDOW_SIZE
            variant_result = variant_model.predict(
                full_jm[start:end].copy(),
                topk=5,
                person_ids=[1],
                image_size=IMAGE_SIZE,
                input_is_joint_motion=True,
            )["persons"][0]
            hand_idx = next(
                index for index, name in enumerate(variant_model.class_names)
                if name.strip().lower() == "hand waving"
            )
            variant_rows.append({
                "window_index": window_index,
                "frames": [start, end - 1],
                "top1": variant_result["action"],
                "top1_confidence": float(variant_result["confidence"]),
                "hand_waving_probability": float(variant_result["probabilities"][hand_idx]),
                "top5": [
                    {
                        "action": row["action"],
                        "class_id": int(row["class_id"]),
                        "confidence": float(row["confidence"]),
                    }
                    for row in variant_result["top_predictions"]
                ],
            })
        diagnostic_matrix[variant_name] = variant_rows

    report = {
        "test": "KTH deterministic 100-frame windows with full-sequence PYSKL Joint Motion",
        "ground_truth": "Hand Waving",
        "random_sampling": False,
        "source": {
            "keypoint_artifact": "kth-yolo-large-pytorch-gcn-report/person_keypoints.npz",
            "track": track_keys,
            "frames": 581,
            "image_size_width_height": list(IMAGE_SIZE),
            "checkpoint": CHECKPOINT.name,
        },
        "preprocessing_order": [
            "PYSKL PreNormalize2D on all 581 frames",
            "PYSKL GenSkeFeat(feats=['jm']) on all 581 frames",
            "slice deterministic contiguous 100-frame windows from the full JM sequence",
            "pass each precomputed 100-frame JM window to model.predict(input_is_joint_motion=True)",
        ],
        "window_starts": WINDOW_STARTS,
        "windows_evaluated": len(all_predictions),
        "windows_matching_ground_truth": sum(row["matches_ground_truth"] for row in all_predictions),
        "summary": {
            "mean_inference_ms": float(np.mean([row["inference_ms"] for row in all_predictions])),
            "wall_seconds": round(time.perf_counter() - started_all, 4),
            "forward_hook_events": len(hook_events),
        },
        "diagnostic_matrix": diagnostic_matrix,
        "tracks": track_reports,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
