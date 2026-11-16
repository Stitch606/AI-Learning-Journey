from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
VIDEO_URL = "https://www.csc.kth.se/cvap/actions/person15_handwaving_d1_uncomp.avi"
VIDEO = ROOT / "artifacts/source_handwaving.avi"
OUT = ROOT / "artifacts" / os.environ.get("RUN_NAME", "ncnn-run")
ARCH = ROOT / "huggingface-space/process/models/ST_GCN++.py"
CHECKPOINT = ROOT / "huggingface-space/process/weights/stgcnpp_jm_ntu60_xsub_hrnet_finetuned.pth"
YOLO_MODEL = "yolo26n-pose.pt"
CONF = 0.25
CHUNK = 30
FRAMES_LIMIT = 300
GCN_WINDOW = 100


def load_arch():
    spec = importlib.util.spec_from_file_location("stgcn_ncnn_runner", ARCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def download_video():
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    if VIDEO.exists() and VIDEO.stat().st_size > 10000:
        return
    req = urllib.request.Request(VIDEO_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response, VIDEO.open("wb") as out:
        while data := response.read(1024 * 1024):
            out.write(data)
    if VIDEO.stat().st_size < 10000:
        raise RuntimeError("Source video download is incomplete")


def read_video():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError("Cannot decode KTH source video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while len(frames) < FRAMES_LIMIT:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < CHUNK:
        raise RuntimeError("Video contains fewer than 30 frames")
    return frames, fps, width, height


def export_models(inference_size):
    model_dir = OUT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir = model_dir / "yolo26n-pose_ncnn_model"
    yolo_start = time.perf_counter()
    if not yolo_dir.exists():
        YOLO(YOLO_MODEL).export(format="ncnn", half=True, imgsz=inference_size, device="cpu")
        exported = ROOT / "yolo26n-pose_ncnn_model"
        if not exported.exists():
            candidates = list(ROOT.glob("*ncnn*"))
            candidates = [p for p in candidates if p.is_dir() and (p / "model.ncnn.param").exists()]
            if not candidates:
                raise RuntimeError("Ultralytics NCNN export did not create an NCNN model directory")
            exported = candidates[0]
        exported.rename(yolo_dir)
    required_yolo_files = (yolo_dir / "model.ncnn.param", yolo_dir / "model.ncnn.bin")
    if not yolo_dir.is_dir() or not all(path.is_file() for path in required_yolo_files):
        raise RuntimeError(
            f"Invalid YOLO NCNN export directory: {yolo_dir}; "
            "expected model.ncnn.param and model.ncnn.bin"
        )

    yolo = YOLO(str(yolo_dir), task="pose")
    probe = np.zeros((inference_size, inference_size, 3), dtype=np.uint8)
    probe_start = time.perf_counter()
    probe_results = yolo.predict(
        probe, imgsz=inference_size, verbose=False, conf=CONF, device="cpu"
    )
    if len(probe_results) != 1:
        raise RuntimeError("YOLO NCNN predict preflight did not return one result")
    if os.environ.get("USE_TRACKER", "1") == "1":
        tracker_results = yolo.track(
            probe, persist=False, tracker="bytetrack.yaml",
            verbose=False, conf=CONF, device="cpu"
        )
        if len(tracker_results) != 1:
            raise RuntimeError("YOLO NCNN tracking preflight did not return one result")
    yolo_preflight_seconds = time.perf_counter() - probe_start
    print(
        f"YOLO NCNN preflight passed: {yolo_dir.name}; "
        f"predict{' + track' if os.environ.get('USE_TRACKER', '1') == '1' else ''}",
        flush=True,
    )
    yolo_export_seconds = time.perf_counter() - yolo_start

    torch.set_num_threads(2)
    mod = load_arch()
    gcn = mod.STGCNPPMultiPerson(
        checkpoint_path=str(CHECKPOINT), device="cpu", max_persons=10
    ).eval()

    class ExportOne(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.source = source
        def forward(self, x):
            x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            logits = self.source._person_clip_logits(x)
            return logits.mean(dim=1).squeeze(0).squeeze(0)

    example = torch.zeros((100, 17, 3), dtype=torch.float32)
    traced_path = model_dir / "stgcnpp_prepared.pt"
    traced = torch.jit.trace(ExportOne(gcn).eval(), example, strict=False)
    traced.save(str(traced_path))
    pnnx_start = time.perf_counter()
    subprocess.run(
        ["pnnx", str(traced_path), "inputshape=[100,17,3]", "device=cpu", "fp16=0"],
        cwd=model_dir, check=True, text=True
    )
    param = model_dir / "stgcnpp_prepared.ncnn.param"
    binary = model_dir / "stgcnpp_prepared.ncnn.bin"
    if not param.exists() or not binary.exists():
        raise RuntimeError("PNNX did not produce ST-GCN++ NCNN param/bin files")
    pnnx_seconds = time.perf_counter() - pnnx_start
    import ncnn
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 2
    if net.load_param(str(param)) != 0 or net.load_model(str(binary)) != 0:
        raise RuntimeError("NCNN could not load converted ST-GCN++")

    parity_rng = np.random.default_rng(606)
    parity_sample = parity_rng.normal(0.0, 0.15, size=(100, 17, 3)).astype(np.float32)
    parity_sample[..., 2] = parity_rng.uniform(0.05, 1.0, size=(100, 17)).astype(np.float32)
    with torch.inference_mode():
        expected_logits = ExportOne(gcn).eval()(torch.from_numpy(parity_sample))
        expected = expected_logits.cpu().numpy()

    parity_start = time.perf_counter()
    extractor = net.create_extractor()
    parity_mat = ncnn.Mat(np.ascontiguousarray(parity_sample))
    code = extractor.input("in0", parity_mat)
    if code != 0:
        raise RuntimeError(f"NCNN GCN parity input failed: {code}")
    code, parity_output = extractor.extract("out0")
    if code != 0:
        raise RuntimeError(f"NCNN GCN parity output failed: {code}")
    actual = np.asarray(parity_output, dtype=np.float32).reshape(-1)
    expected = expected.reshape(-1)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise RuntimeError(
            f"Invalid NCNN GCN parity output: actual={actual.shape}, expected={expected.shape}"
        )
    parity_error = np.abs(actual - expected)
    parity_max_abs = float(parity_error.max())
    parity_mean_abs = float(parity_error.mean())
    if not np.allclose(actual, expected, atol=2e-3, rtol=2e-3):
        raise RuntimeError(
            "NCNN/PyTorch ST-GCN++ parity failed: "
            f"max_abs={parity_max_abs:.6g}, mean_abs={parity_mean_abs:.6g}"
        )
    parity_seconds = time.perf_counter() - parity_start
    print(
        "ST-GCN++ NCNN parity passed: "
        f"max_abs={parity_max_abs:.6g}, mean_abs={parity_mean_abs:.6g}",
        flush=True,
    )
    return {
        "yolo": yolo,
        "gcn": gcn,
        "ncnn": net,
        "export_times": {
            "yolo_ncnn_export_seconds": round(yolo_export_seconds, 4),
            "yolo_ncnn_preflight_seconds": round(yolo_preflight_seconds, 4),
            "gcn_pnnx_conversion_seconds": round(pnnx_seconds, 4),
            "gcn_ncnn_parity_seconds": round(parity_seconds, 4),
            "gcn_ncnn_parity_max_abs_error": parity_max_abs,
            "gcn_ncnn_parity_mean_abs_error": parity_mean_abs,
        },
    }


def gcn_ncnn_predict(net, prepared):
    import ncnn
    extractor = net.create_extractor()
    mat = ncnn.Mat(np.ascontiguousarray(prepared.astype(np.float32)))
    code = extractor.input("in0", mat)
    if code != 0:
        raise RuntimeError(f"NCNN GCN input failed: {code}")
    code, output = extractor.extract("out0")
    if code != 0:
        raise RuntimeError(f"NCNN GCN output failed: {code}")
    logits = np.asarray(output, dtype=np.float32).reshape(-1)
    if logits.size != 60 or not np.isfinite(logits).all():
        raise RuntimeError(f"Invalid NCNN GCN logits shape/values: {logits.shape}")
    shifted = logits - float(logits.max())
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / float(exp_logits.sum())
    probability_sum = float(probabilities.sum())
    if not np.isclose(probability_sum, 1.0, atol=1e-6):
        raise RuntimeError(f"NCNN GCN output is not a probability distribution: sum={probability_sum:.8f}")
    top = int(probabilities.argmax())
    return {"class_id": top, "confidence": float(probabilities[top]),
            "probabilities": probabilities.tolist(), "logits": logits.tolist()}


def resize_frame(frame, enabled, size):
    if not enabled:
        return frame
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def scheduled_positions(real_count):
    # Deterministic, evenly distributed real YOLO samples in each 30-frame block.
    return set((i * CHUNK) // real_count for i in range(real_count))


def flow_pose(prev_gray, gray, pose):
    xy = pose[:, :2].astype(np.float32).reshape(-1, 1, 2)
    conf = pose[:, 2].copy()
    valid = conf > 0.05
    if int(valid.sum()) < 3:
        return np.zeros_like(pose)
    points = xy[valid]
    moved, status, errors = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )
    result = np.zeros_like(pose)
    ids = np.flatnonzero(valid)
    if moved is None or status is None:
        return result
    for j, idx in enumerate(ids):
        if status[j, 0] and np.isfinite(moved[j, 0]).all():
            result[idx, :2] = moved[j, 0]
            result[idx, 2] = conf[idx] * 0.98
    return result


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    mode = os.environ["MODE"]
    real_count = int(os.environ["REAL_PER_30"])
    resize = os.environ.get("RESIZE_INPUT", os.environ.get("RESIZE_256", "0")) == "1"
    resize_size = int(os.environ.get("RESIZE_SIZE", "256"))
    inference_size = resize_size if resize else 640
    use_tracker = os.environ.get("USE_TRACKER", "1") == "1"
    use_flow = os.environ.get("USE_FLOW", "1") == "1"

    download_video()
    frames, source_fps, original_w, original_h = read_video()
    prepared_models = export_models(inference_size)
    yolo = prepared_models["yolo"]
    gcn = prepared_models["gcn"]
    net = prepared_models["ncnn"]

    prep_times = defaultdict(float)
    yolo_times = []
    flow_times = []
    gcn_times = []
    records = defaultdict(lambda: deque(maxlen=GCN_WINDOW))
    latest_pose = {}
    latest_gray = None
    idless_slot = 1
    detection_count = 0
    flow_success = 0
    flow_rejected = 0
    frame_detection_counts = []
    final_predictions = {}
    inference_start = time.perf_counter()
    positions = scheduled_positions(real_count) if real_count else set()

    for block_start in range(0, len(frames) - (len(frames) % CHUNK), CHUNK):
        block = frames[block_start:block_start + CHUNK]
        anchors = positions if real_count < CHUNK else set(range(CHUNK))
        block_poses = {}

        for offset, original in enumerate(block):
            index = block_start + offset
            frame = resize_frame(original, resize, resize_size)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if offset in anchors:
                start = time.perf_counter()
                if use_tracker:
                    results = yolo.track(
                        frame, persist=True, tracker="bytetrack.yaml",
                        verbose=False, conf=CONF, device="cpu"
                    )
                else:
                    results = yolo.predict(
                        frame, verbose=False, conf=CONF, device="cpu"
                    )
                yolo_times.append(time.perf_counter() - start)
                result = results[0]
                poses = result.keypoints.data.cpu().numpy() if result.keypoints is not None else np.empty((0,17,3))
                boxes = result.boxes
                ids = boxes.id.cpu().numpy().astype(int).tolist() if use_tracker and boxes is not None and boxes.id is not None else []
                if not use_tracker:
                    # The KTH source is a single-person action clip: stable largest-box slot, no tracker.
                    if boxes is not None and len(boxes):
                        areas = (boxes.xyxy[:,2] - boxes.xyxy[:,0]) * (boxes.xyxy[:,3] - boxes.xyxy[:,1])
                        chosen = int(areas.argmax())
                        poses = poses[chosen:chosen+1]
                    else:
                        poses = np.empty((0,17,3))
                    ids = [1] if len(poses) else []
                for pi, pose in enumerate(poses):
                    if pi >= len(ids):
                        continue
                    tid = int(ids[pi])
                    block_poses.setdefault(tid, {})[offset] = pose.astype(np.float32)
                    latest_pose[tid] = pose.astype(np.float32)
                    detection_count += 1
                frame_detection_counts.append(len(poses))
            else:
                start = time.perf_counter()
                if use_flow and latest_gray is not None and latest_pose:
                    for tid, pose in list(latest_pose.items()):
                        moved = flow_pose(latest_gray, gray, pose)
                        if int((moved[:,2] > 0.05).sum()) >= 5:
                            block_poses.setdefault(tid, {})[offset] = moved
                            latest_pose[tid] = moved
                            flow_success += 1
                        else:
                            flow_rejected += 1
                flow_times.append(time.perf_counter() - start)
                frame_detection_counts.append(len(latest_pose))
            latest_gray = gray

        # Ensure every frame contributes a pose: interpolate only between valid anchor/flow samples
        # for the same ByteTrack ID; missing poses remain zero-confidence and are rejected by GCN.
        all_ids = sorted(set(records) | {tid for values in block_poses.values() for tid in values})
        for tid in all_ids:
            block_poses.setdefault(tid, {})
            known = sorted(block_poses[tid])
            for offset in range(CHUNK):
                if offset in block_poses[tid]:
                    continue
                left = max((k for k in known if k < offset), default=None)
                right = min((k for k in known if k > offset), default=None)
                if left is not None and right is not None:
                    alpha = (offset - left) / (right - left)
                    p0, p1 = block_poses[tid][left], block_poses[tid][right]
                    interp = p0 * (1 - alpha) + p1 * alpha
                    interp[:,2] = np.minimum(p0[:,2], p1[:,2]) * 0.9
                    block_poses[tid][offset] = interp.astype(np.float32)

            seq = records[tid]
            for offset in range(CHUNK):
                pose = block_poses[tid].get(offset)
                if pose is None:
                    seq.append(np.zeros((17,3), dtype=np.float32))
                else:
                    seq.append(pose)
            if len(seq) >= GCN_WINDOW:
                sequence = np.stack(list(seq)[-GCN_WINDOW:])
                if resize:
                    scale = min(resize_size / original_w, resize_size / original_h)
                    resized_w = max(1, round(original_w * scale))
                    resized_h = max(1, round(original_h * scale))
                    pad_x = (resize_size - resized_w) // 2
                    pad_y = (resize_size - resized_h) // 2
                    sequence[:, :, 0] = (sequence[:, :, 0] - pad_x) / scale
                    sequence[:, :, 1] = (sequence[:, :, 1] - pad_y) / scale
                try:
                    valid_frames = int((sequence[:,:,2].sum(axis=1) > 0).sum())
                    if valid_frames < 60:
                        continue
                    prepared = gcn._prepare_single_person(
                        sequence, image_size=(original_w, original_h)
                    )
                    start = time.perf_counter()
                    pred = gcn_ncnn_predict(net, prepared)
                    gcn_times.append(time.perf_counter() - start)
                    class_names = gcn.class_names
                    pred["action"] = class_names[pred["class_id"]]
                    pred["valid_frames"] = valid_frames
                    final_predictions[str(tid)] = pred
                except Exception as exc:
                    final_predictions[str(tid)] = {"error": f"{type(exc).__name__}: {exc}"}

    total_elapsed = time.perf_counter() - inference_start
    processed = (len(frames) // CHUNK) * CHUNK
    report = {
        "experiment": mode,
        "video_url": VIDEO_URL,
        "video_frames_decoded": len(frames),
        "frames_in_complete_30_frame_blocks": processed,
        "frames_discarded_tail": len(frames) - processed,
        "source_fps": source_fps,
        "source_resolution": [original_w, original_h],
        "input_resize": f"letterbox_{resize_size}x{resize_size}" if resize else "native",
        "yolo_inference_size": inference_size,
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "real_yolo_frames_per_30": real_count,
        "optical_flow_frames_per_30": CHUNK - real_count,
        "real_frame_positions_zero_based": sorted(positions),
        "tracker": "ByteTrack" if use_tracker else "disabled; largest detected person slot per frame",
        "optical_flow": use_flow,
        "all_frames_submitted_to_pose_sequence": True,
        "yolo_inference_calls": len(yolo_times),
        "optical_flow_calls": len(flow_times),
        "optical_flow_person_frame_updates": flow_success,
        "optical_flow_rejected_updates": flow_rejected,
        "person_detections": detection_count,
        "gcn_ncnn_calls": len(gcn_times),
        "gcn_ncnn_weight_storage": "FP32 (PNNX fp16=0)",
        "timing_seconds": {
            "yolo_total": round(sum(yolo_times), 5),
            "yolo_mean": round(float(np.mean(yolo_times)), 6) if yolo_times else 0,
            "optical_flow_total": round(sum(flow_times), 5),
            "gcn_total": round(sum(gcn_times), 5),
            "gcn_mean": round(float(np.mean(gcn_times)), 6) if gcn_times else 0,
            "pipeline_wall": round(total_elapsed, 5),
            **prepared_models["export_times"],
        },
        "throughput_fps": round(processed / total_elapsed, 4) if total_elapsed else 0,
        "predictions_by_person": final_predictions,
        "hardware": {
            "runner": os.environ.get("RUNNER_NAME", "GitHub-hosted ubuntu-latest"),
            "torch_threads": torch.get_num_threads(),
            "ncnn_threads": 2,
        },
        "notes": [
            "YOLO Pose and ST-GCN++ inference both run through NCNN.",
            "GCN preprocessing and Joint-Motion construction are NumPy operations outside the NCNN graph.",
            "Optical Flow propagates the last valid pose for each current ByteTrack ID; failed estimates are not promoted to valid poses.",
            "Action 3 intentionally has no temporal tracker; largest-box slot is only meaningful for the supplied single-person KTH sample.",
            "The sample is one labeled action clip and is not a statistically meaningful accuracy benchmark."
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not final_predictions:
        raise RuntimeError("No valid 100-frame person sequence reached the NCNN GCN.")
    if any("error" in p for p in final_predictions.values()):
        raise RuntimeError("At least one GCN NCNN inference failed; see report.json")


if __name__ == "__main__":
    run()
