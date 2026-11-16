from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "huggingface-space"
sys.path.insert(0, str(SPACE))

VIDEO_URL = "https://www.csc.kth.se/cvap/actions/person15_handwaving_d1_uncomp.avi"
VIDEO_PATH = Path("artifacts/source_handwaving.avi")
OUTPUT_VIDEO = Path("artifacts/combined_annotated.mp4")
REPORT_PATH = Path("artifacts/evaluation_report.json")
YOLO_MODEL = "yolo26m-pose.pt"
CHECKPOINT = SPACE / "process/weights/stgcnpp_jm_ntu60_xsub_hrnet_finetuned.pth"
ARCH_PATH = SPACE / "process/models/ST_GCN++.py"
YOLO_CONF = 0.25
MAX_FRAMES = 100


def load_gcn():
    spec = importlib.util.spec_from_file_location("project_stgcnpp", ARCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load architecture: {ARCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STGCNPPMultiPerson(
        checkpoint_path=str(CHECKPOINT),
        device="cpu",
        max_persons=10,
    )


def download_video():
    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        VIDEO_URL, headers={"User-Agent": "Mozilla/5.0 action-evaluation/1.0"}
    )
    with urllib.request.urlopen(request, timeout=90) as response, VIDEO_PATH.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if VIDEO_PATH.stat().st_size < 10_000:
        raise RuntimeError("The public sample video download is unexpectedly small.")


def read_video():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot decode the downloaded KTH AVI video.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while len(frames) < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("The downloaded video contains no decodable frames.")
    return frames, fps, width, height


def evaluate_yolo(frames):
    model = YOLO(YOLO_MODEL)
    sequences = {}
    per_frame_counts = []
    visible_joint_values = []
    keypoint_confidences = []
    detected_instances = 0
    started = time.perf_counter()

    for frame_index, frame in enumerate(frames):
        result_list = model.track(
            frame, persist=True, verbose=False, conf=YOLO_CONF, device="cpu"
        )
        result = result_list[0] if result_list else None
        count = 0
        if result is not None and result.boxes is not None and result.keypoints is not None:
            points = result.keypoints.data.detach().cpu().numpy()
            ids_tensor = result.boxes.id
            ids = (
                ids_tensor.detach().cpu().numpy().astype(int).tolist()
                if ids_tensor is not None
                else []
            )
            for index, track_id in enumerate(ids):
                if index >= len(points):
                    continue
                if track_id not in sequences:
                    sequences[track_id] = np.zeros((len(frames), 17, 3), dtype=np.float32)
                person_points = points[index].astype(np.float32, copy=True)
                sequences[track_id][frame_index] = person_points
                visible = person_points[:, 2] >= YOLO_CONF
                visible_joint_values.append(int(visible.sum()))
                if visible.any():
                    keypoint_confidences.extend(person_points[visible, 2].tolist())
                count += 1
        per_frame_counts.append(count)
        detected_instances += count

    elapsed = time.perf_counter() - started
    report = {
        "frames": len(frames),
        "inference_calls": len(frames),
        "detected_person_instances": detected_instances,
        "unique_track_ids": len(sequences),
        "mean_person_detections_per_frame": round(float(np.mean(per_frame_counts)), 3),
        "frames_with_person_detection": int(sum(v > 0 for v in per_frame_counts)),
        "mean_visible_keypoints_per_detection": (
            round(float(np.mean(visible_joint_values)), 3) if visible_joint_values else 0.0
        ),
        "mean_visible_keypoint_confidence": (
            round(float(np.mean(keypoint_confidences)), 4) if keypoint_confidences else 0.0
        ),
        "elapsed_seconds": round(elapsed, 3),
        "throughput_fps": round(len(frames) / elapsed, 3) if elapsed else 0.0,
        "ground_truth_box_or_pose_annotations_available": False,
        "note": "Detection/pose quality is reported operationally; AP/precision/recall require box/keypoint ground truth.",
    }
    return model, sequences, report


def evaluate_gcn(gcn, sequences, image_size):
    people = []
    for track_id, sequence in sorted(sequences.items()):
        try:
            prediction = gcn.predict(
                sequence,
                topk=5,
                person_ids=[track_id],
                image_size=image_size,
            )["persons"][0]
            people.append({
                "track_id": int(track_id),
                "frames_in_clip": int(len(sequence)),
                "frames_with_pose": int(np.count_nonzero(sequence[:, :, 2].sum(axis=1) > 0)),
                "top1": prediction["action"],
                "top1_class_id": prediction["class_id"],
                "top1_confidence": round(float(prediction["confidence"]), 6),
                "top5": [
                    {
                        "action": item["action"],
                        "class_id": item["class_id"],
                        "confidence": round(float(item["confidence"]), 6),
                    }
                    for item in prediction["top_predictions"]
                ],
                "matches_video_label": prediction["action"].strip().lower() == "hand waving",
            })
        except Exception as error:
            people.append({
                "track_id": int(track_id),
                "error": f"{type(error).__name__}: {error}",
            })
    successful = [person for person in people if "top1" in person]
    return {
        "people_evaluated": len(successful),
        "people_rejected_or_failed": len(people) - len(successful),
        "ground_truth_action": "hand waving",
        "top1_label_matches": sum(bool(person["matches_video_label"]) for person in successful),
        "per_person": people,
        "note": "This is a labeled single-action sample; one clip is a diagnostic case, not a statistically meaningful accuracy estimate.",
    }


def evaluate_combined(frames, fps, gcn):
    from process.streaming_inference import StreamingInference

    engine = StreamingInference(yolo=YOLO(YOLO_MODEL), mlti=gcn)
    # Offline video decoding can run slower than real time on a CPU runner.
    # Keep tracks alive for the duration of this file instead of applying the
    # live-camera wall-clock disappearance timeout.
    engine.manager.disappear_timeout = 3600.0
    sample_interval = max(1, round(fps / 10.0))
    writer = None
    yolo_calls = 0
    flow_calls = 0
    detections_per_frame = []
    started = time.perf_counter()

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    for frame_index, bgr in enumerate(frames):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame_index % sample_interval == 0:
            result = engine.process_frame(rgb)
            yolo_calls += 1
        else:
            result = engine.process_flow_frame(rgb)
            flow_calls += 1

        detections_per_frame.append(len(result.get("people", [])))
        encoded = result.get("annotated_frame")
        if encoded and encoded.startswith("data:image/"):
            image_bytes = base64.b64decode(encoded.split(",", 1)[1])
            annotated = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if annotated is None:
                annotated = bgr
        else:
            annotated = bgr

        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(
                str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError("Could not create the annotated MP4 output.")
        writer.write(annotated)

    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - started
    final_people = [
        {
            "track_id": int(track_id),
            "action": prediction["action"],
            "confidence": round(float(prediction["confidence"]), 6),
        }
        for track_id, prediction in sorted(engine.predictions.items())
    ]
    return {
        "frames": len(frames),
        "yolo_calls": yolo_calls,
        "optical_flow_calls": flow_calls,
        "mean_active_people_per_frame": round(float(np.mean(detections_per_frame)), 3),
        "elapsed_seconds": round(elapsed, 3),
        "throughput_fps": round(len(frames) / elapsed, 3) if elapsed else 0.0,
        "per_person_predictions": final_people,
        "annotated_video": str(OUTPUT_VIDEO),
    }


def main():
    torch.set_num_threads(2)
    download_video()
    frames, fps, width, height = read_video()
    if len(frames) != MAX_FRAMES:
        print(f"INFO: source yielded {len(frames)} frames (target was {MAX_FRAMES}).")

    print("Loading current project ST-GCN++ checkpoint...")
    gcn = load_gcn()

    print("Running YOLO-only evaluation...")
    _, sequences, yolo_report = evaluate_yolo(frames)

    print("Running independent per-track GCN evaluation...")
    gcn_report = evaluate_gcn(gcn, sequences, (width, height))

    print("Running the combined production streaming path...")
    combined_report = evaluate_combined(frames, fps, gcn)

    report = {
        "video": {
            "source_url": VIDEO_URL,
            "dataset": "KTH Actions",
            "ground_truth_action": "hand waving",
            "frames_processed": len(frames),
            "fps": fps,
            "resolution": [width, height],
            "duration_seconds": round(len(frames) / fps, 3) if fps else None,
            "download_bytes": VIDEO_PATH.stat().st_size,
        },
        "models": {
            "yolo": YOLO_MODEL,
            "stgcn_architecture": str(ARCH_PATH),
            "stgcn_checkpoint": str(CHECKPOINT),
            "device": "cpu",
        },
        "yolo_only": yolo_report,
        "gcn_only": gcn_report,
        "combined_streaming": combined_report,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
