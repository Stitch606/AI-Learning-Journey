from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/ncnn_pipeline"))
import real_video_ncnn_experiments as exp


def main():
    torch.set_num_threads(2)
    exp.OUT = ROOT / "artifacts" / "diagnose-direct-256"
    exp.download_video()
    cap = cv2.VideoCapture(str(exp.VIDEO))
    if not cap.isOpened():
        raise RuntimeError("Cannot open source KTH video")
    frames = []
    while len(frames) < 120:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != 120:
        raise RuntimeError(f"Expected 120 source frames, got {len(frames)}")

    models = exp.export_models(256)
    yolo, gcn, net = models["yolo"], models["gcn"], models["ncnn"]
    sequence = deque(maxlen=100)
    original_h, original_w = frames[0].shape[:2]
    start = time.perf_counter()

    for frame in frames:
        resized = exp.resize_frame(frame, True, 256)
        results = yolo.predict(resized, verbose=False, conf=exp.CONF, device="cpu")
        result = results[0]
        if result.boxes is not None and len(result.boxes) and result.keypoints is not None:
            areas = (result.boxes.xyxy[:, 2] - result.boxes.xyxy[:, 0]) * (result.boxes.xyxy[:, 3] - result.boxes.xyxy[:, 1])
            idx = int(areas.argmax())
            pose = result.keypoints.data[idx].cpu().numpy().astype(np.float32, copy=True)
            sequence.append(pose)
        else:
            sequence.append(np.zeros((17, 3), dtype=np.float32))

    seq = np.stack(list(sequence))
    scale = min(256 / original_w, 256 / original_h)
    resized_w = max(1, round(original_w * scale))
    resized_h = max(1, round(original_h * scale))
    pad_x = (256 - resized_w) // 2
    pad_y = (256 - resized_h) // 2
    seq[:, :, 0] = (seq[:, :, 0] - pad_x) / scale
    seq[:, :, 1] = (seq[:, :, 1] - pad_y) / scale
    prepared = gcn._prepare_single_person(seq, image_size=(original_w, original_h))

    x = torch.from_numpy(prepared).reshape(1, 1, 1, 100, 17, 3)
    with torch.inference_mode():
        logits = gcn._person_clip_logits(x)
        torch_logits = logits.mean(dim=1).squeeze(0).squeeze(0).cpu().numpy()
        torch_probs = torch.softmax(torch.from_numpy(torch_logits), dim=-1).numpy()

    ncnn_result = exp.gcn_ncnn_predict(net, prepared)
    ncnn_logits = np.asarray(ncnn_result["logits"], dtype=np.float32)
    ncnn_probs = np.asarray(ncnn_result["probabilities"], dtype=np.float32)
    report = {
        "video": "KTH person15_handwaving_d1_uncomp.avi",
        "frames_used": 120,
        "effective_gcn_window": 100,
        "resize": "letterbox_256x256",
        "ncnn_weight_storage": "FP32 (PNNX fp16=0)",
        "ncnn_output": "raw logits followed by stable NumPy softmax",
        "valid_pose_frames": int((seq[:, :, 2].sum(axis=1) > 0).sum()),
        "prepared_min": float(prepared.min()),
        "prepared_max": float(prepared.max()),
        "prepared_mean_abs": float(np.abs(prepared).mean()),
        "torch_probability_sum": float(torch_probs.sum()),
        "torch_top1_class": int(torch_probs.argmax()),
        "torch_top1_action": gcn.class_names[int(torch_probs.argmax())],
        "torch_top1_confidence": float(torch_probs.max()),
        "ncnn_probability_sum": float(ncnn_probs.sum()),
        "ncnn_top1_class": int(ncnn_probs.argmax()),
        "ncnn_top1_action": gcn.class_names[int(ncnn_probs.argmax())],
        "ncnn_top1_confidence": float(ncnn_probs.max()),
        "max_abs_logit_difference": float(np.max(np.abs(torch_logits - ncnn_logits))),
        "mean_abs_logit_difference": float(np.mean(np.abs(torch_logits - ncnn_logits))),
        "max_abs_probability_difference": float(np.max(np.abs(torch_probs - ncnn_probs))),
        "mean_abs_probability_difference": float(np.mean(np.abs(torch_probs - ncnn_probs))),
        "wall_seconds": round(time.perf_counter() - start, 4),
    }
    (ROOT / "artifacts/diagnose-direct-256").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts/diagnose-direct-256/report.json").write_text(json.dumps(report, indent=2))
    np.save(ROOT / "artifacts/diagnose-direct-256/prepared_input.npy", prepared)
    print(json.dumps(report, indent=2))
    if not np.isclose(ncnn_probs.sum(), 1.0, atol=1e-3):
        raise RuntimeError("NCNN output is not a valid probability distribution")
    if not np.allclose(torch_probs, ncnn_probs, atol=2e-3, rtol=2e-3):
        raise RuntimeError("Real-input PyTorch/NCNN parity failed")


if __name__ == "__main__":
    main()
