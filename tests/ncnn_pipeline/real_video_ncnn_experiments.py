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


def export_models():
    model_dir = OUT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir = model_dir / "yolo26n_pose_ncnn"
    yolo_start = time.perf_counter()
    if not yolo_dir.exists():
        YOLO(YOLO_MODEL).export(format="ncnn", half=True, imgsz=640, device="cpu")
        exported = ROOT / "yolo26n-pose_ncnn_model"
        if not exported.exists():
            candidates = list(ROOT.glob("*ncnn*"))
            candidates = [p for p in candidates if p.is_dir() and (p / "model.ncnn.param").exists()]
            if not candidates:
                raise RuntimeError("Ultralytics NCNN export did not create an NCNN model directory")
            exported = candidates[0]
        exported.rename(yolo_dir)
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
            return torch.softmax(logits.mean(dim=1).squeeze(0).squeeze(0), dim=-1)

    example = torch.zeros((100, 17, 3), dtype=torch.float32)
    traced_path = model_dir / "stgcnpp_prepared.pt"
    traced = torch.jit.trace(ExportOne(gcn).eval(), example, strict=False)
    traced.save(str(traced_path))
    pnnx_start = time.perf_counter()
    subprocess.run(
        ["pnnx", str(traced_path), "inputshape=[100,17,3]", "device=cpu"],
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
    return {
        "yolo": YOLO(str(yolo_dir)),
        "gcn": gcn,
        "ncnn": net,
        "export_times": {
            "yolo_ncnn_export_seconds": round(yolo_export_seconds, 4),
            "gcn_pnnx_conversion_seconds": round(pnnx_seconds, 4),
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
    probabilities = np.asarray(output, dtype=np.float32).reshape(-1)
    if probabilities.size != 60 or not np.isfinite(probabilities).all():
        raise RuntimeError(f"Invalid NCNN GCN output shape/values: {probabilities.shape}")
    top = int(probabilities.argmax())
    return {"class_id": top, "confidence": float(probabilities[top]),
            "probabilities": probabilities.tolist()}


def resize_frame(frame, enabled):
    if not enabled:
        return frame
    h, w = frame.shape[:2]
    scale = min(256 / w, 256 / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((256, 256, 3), dtype=np.uint8)
    x0, y0 = (256 - nw) // 2, (256 - nh) // 2
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
    resize = os.environ.get("RESIZE_256", "0") == "1"
    use_tracker = os.environ.get("USE_TRACKER", "1") == "1"
    use_flow = os.environ.get("USE_FLOW", "1") == "1"

    download_video()
    frames, source_fps, original_w, original_h = read_video()
    prepared_models = export_models()
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
    latest_frame_index = None
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
        prev_gray = None
        active_ids = set()

        for offset, original in enumerate(block):
            index = block_start + offset
            frame = resize_frame(original, resize)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if offset in anchors:
                start = time.perf_counter()
                results = yolo.track(
                    frame, persist=use_tracker, tracker="bytetrack.yaml" if use_tracker else None,
                    verbose=False, conf=CONF, device="cpu"
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
                    if resize:
                        # Undo letterbox to original pixel coordinates before GCN preprocessing.
                        h0, w0 = original.shape[:2]
                        scale = min(256 / w0, 256 / h0)
                        nw, nh = max(1, round(w0 * scale)), max(1, round(h0 * scale))
                        x0, y0 = (256 - nw) // 2, (256 - nh) // 2
                        pose = pose.copy()
                        pose[:,0] = (pose[:,0] - x0) / scale
                        pose[:,1] = (pose[:,1] - y0) / scale
                    block_poses.setdefault(tid, {})[offset] = pose.astype(np.float32)
                    latest_pose[tid] = pose.astype(np.float32)
                    active_ids.add(tid)
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
            latest_frame_index = index

        # Ensure every frame contributes a pose: interpolate only between valid anchor/flow samples
        # for the same ByteTrack ID; missing poses remain zero-confidence and are rejected by GCN.
        all_ids = sorted({tid for values in block_poses.values() for tid in values})
        for tid in all_ids:
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
        "input_resize": "letterbox_256x256" if resize else "native",
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
