from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from GCN.STGCNPP.model_multi_person import STGCNPPMultiPerson


VIDEO_URL = "https://www.csc.kth.se/cvap/actions/person15_handwaving_d1_uncomp.avi"
VIDEO_PATH = ROOT / "artifacts" / "kth-yolo-large-pytorch-gcn" / "source_handwaving.avi"
OUTPUT_DIR = ROOT / "artifacts" / "kth-yolo-large-pytorch-gcn"
REPORT_PATH = OUTPUT_DIR / "report.json"
KEYPOINTS_PATH = OUTPUT_DIR / "person_keypoints.npz"
CHECKPOINT = ROOT / "GCN" / "checkpoints" / "stgcnpp_jm_ntu60_xsub_hrnet_finetuned.pth"
YOLO_MODEL = "yolo26l-pose.pt"
YOLO_CONF = 0.25
YOLO_IMGSZ = 640
MAX_PERSONS = 10


def download_video() -> None:
    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if VIDEO_PATH.is_file() and VIDEO_PATH.stat().st_size > 10_000:
        return
    request = urllib.request.Request(
        VIDEO_URL, headers={"User-Agent": "Mozilla/5.0 KTH-action-test/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, VIDEO_PATH.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    if VIDEO_PATH.stat().st_size <= 10_000:
        raise RuntimeError("KTH source video download is incomplete")


def load_models():
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Fine-tuned GCN checkpoint is missing: {CHECKPOINT}")

    checkpoint_data = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    metadata = {
        "epoch": checkpoint_data.get("epoch") if isinstance(checkpoint_data, dict) else None,
        "best_validation_accuracy": (
            checkpoint_data.get("best_validation_accuracy")
            if isinstance(checkpoint_data, dict) else None
        ),
        "validation": checkpoint_data.get("validation") if isinstance(checkpoint_data, dict) else None,
        "fine_tuned": checkpoint_data.get("fine_tuned") if isinstance(checkpoint_data, dict) else None,
    }

    gcn = STGCNPPMultiPerson(
        checkpoint_path=str(CHECKPOINT),
        device="cpu",
        max_persons=MAX_PERSONS,
        strict=True,
    ).eval()

    yolo = YOLO(YOLO_MODEL)
    return yolo, gcn, metadata


def run_yolo_on_original_frames(yolo):
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not decode the KTH Hand Waving video.")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sequences: dict[int, list[np.ndarray]] = {}
    detection_counts = []
    yolo_seconds = []
    detected_instances = 0
    frames_processed = 0
    zero_pose = np.zeros((17, 3), dtype=np.float32)
    started = time.perf_counter()

    while True:
        ok, original_frame = cap.read()
        if not ok:
            break

        frame_index = frames_processed
        for sequence in sequences.values():
            sequence.append(zero_pose.copy())

        inference_start = time.perf_counter()
        results = yolo.track(
            original_frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            device="cpu",
            verbose=False,
        )
        yolo_seconds.append(time.perf_counter() - inference_start)

        result = results[0] if results else None
        frame_poses = 0
        if (
            result is not None
            and result.keypoints is not None
            and result.boxes is not None
            and len(result.keypoints.data) > 0
        ):
            poses = result.keypoints.data.detach().cpu().numpy().astype(np.float32)
            ids_tensor = result.boxes.id
            if ids_tensor is not None and len(ids_tensor) == len(poses):
                track_ids = ids_tensor.detach().cpu().numpy().astype(int).tolist()
                selected = list(range(len(poses)))
            else:
                areas = (
                    (result.boxes.xyxy[:, 2] - result.boxes.xyxy[:, 0])
                    * (result.boxes.xyxy[:, 3] - result.boxes.xyxy[:, 1])
                ).detach().cpu().numpy()
                selected_index = int(np.argmax(areas))
                track_ids = [1]
                selected = [selected_index]

            for index, track_id in zip(selected, track_ids):
                if track_id not in sequences:
                    sequences[track_id] = [
                        zero_pose.copy() for _ in range(frame_index + 1)
                    ]
                sequences[track_id][-1] = poses[index].copy()
                frame_poses += 1
                detected_instances += 1

        detection_counts.append(frame_poses)
        frames_processed += 1

        if frames_processed % 50 == 0:
            print(
                f"YOLO progress: {frames_processed}/{declared_frames or '?'} frames; "
                f"tracks={len(sequences)}",
                flush=True,
            )

    cap.release()
    elapsed = time.perf_counter() - started
    if frames_processed == 0:
        raise RuntimeError("The source video contained no decodable frames.")
    if not sequences:
        raise RuntimeError("YOLO Large did not detect a person in any source frame.")

    return sequences, {
        "source_fps": source_fps,
        "source_resolution": [width, height],
        "declared_frame_count": declared_frames,
        "frames_processed": frames_processed,
        "yolo_inference_calls": len(yolo_seconds),
        "detected_person_instances": detected_instances,
        "unique_track_ids": len(sequences),
        "mean_detections_per_frame": round(float(np.mean(detection_counts)), 4),
        "frames_with_person_detection": int(sum(value > 0 for value in detection_counts)),
        "yolo_total_seconds": round(sum(yolo_seconds), 4),
        "yolo_mean_ms": round(float(np.mean(yolo_seconds)) * 1000, 4),
        "yolo_fps": round(frames_processed / sum(yolo_seconds), 4) if sum(yolo_seconds) else 0.0,
        "detection_loop_wall_seconds": round(elapsed, 4),
    }


def run_gcn(gcn, sequences, image_size):
    predictions = []
    gcn_seconds = []

    for track_id, frames in sorted(sequences.items()):
        sequence = np.stack(frames).astype(np.float32, copy=False)
        valid_frames = int(np.count_nonzero(sequence[:, :, 2].sum(axis=1) > 0))
        if valid_frames < 60:
            predictions.append({
                "track_id": int(track_id),
                "frames_in_video": int(len(sequence)),
                "frames_with_pose": valid_frames,
                "status": "insufficient_pose_frames",
            })
            continue

        started = time.perf_counter()
        output = gcn.predict(
            sequence,
            topk=5,
            person_ids=[int(track_id)],
            image_size=image_size,
        )
        elapsed = time.perf_counter() - started
        gcn_seconds.append(elapsed)
        person = output["persons"][0]

        predictions.append({
            "track_id": int(track_id),
            "frames_in_video": int(len(sequence)),
            "frames_with_pose": valid_frames,
            "top1": person["action"],
            "top1_class_id": int(person["class_id"]),
            "top1_confidence": round(float(person["confidence"]), 8),
            "matches_ground_truth": person["action"].strip().lower() == "hand waving",
            "top5": [
                {
                    "action": item["action"],
                    "class_id": int(item["class_id"]),
                    "confidence": round(float(item["confidence"]), 8),
                }
                for item in person["top_predictions"]
            ],
            "gcn_inference_ms": round(elapsed * 1000, 4),
        })

        print(
            f"GCN track={track_id}: {person['action']} "
            f"({person['confidence']:.6f})",
            flush=True,
        )

    successful = [item for item in predictions if "top1" in item]
    return predictions, {
        "tracks_received": len(sequences),
        "tracks_evaluated": len(successful),
        "tracks_skipped": len(predictions) - len(successful),
        "top1_matches": sum(bool(item["matches_ground_truth"]) for item in successful),
        "gcn_total_seconds": round(sum(gcn_seconds), 4),
        "gcn_mean_ms": round(float(np.mean(gcn_seconds)) * 1000, 4) if gcn_seconds else 0.0,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    print("Loading the exact 84.81% fine-tuned GCN checkpoint...", flush=True)
    yolo, gcn, checkpoint_metadata = load_models()
    print("Models loaded. Running YOLO Large on every original video frame...", flush=True)

    video_request = urllib.request.Request(
        VIDEO_URL, headers={"User-Agent": "Mozilla/5.0 KTH-action-test/1.0"}
    )
    if not VIDEO_PATH.exists():
        with urllib.request.urlopen(video_request, timeout=120) as response, VIDEO_PATH.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)

    sequences, yolo_report = run_yolo_on_original_frames(yolo)
    width, height = yolo_report["source_resolution"]

    print("Passing each complete raw YOLO track to native PyTorch ST-GCN++...", flush=True)
    predictions, gcn_report = run_gcn(gcn, sequences, (width, height))

    np.savez(
        KEYPOINTS_PATH,
        **{f"track_{track_id}": np.stack(frames) for track_id, frames in sequences.items()},
    )

    report = {
        "test": "KTH Hand Waving: YOLO Large + original PyTorch ST-GCN++",
        "ground_truth_action": "hand waving",
        "video": {
            "url": VIDEO_URL,
            "local_file": str(VIDEO_PATH.relative_to(ROOT)),
            "bytes": VIDEO_PATH.stat().st_size,
            **{
                key: yolo_report[key]
                for key in (
                    "source_fps", "source_resolution", "declared_frame_count",
                    "frames_processed"
                )
            },
        },
        "models": {
            "yolo": YOLO_MODEL,
            "yolo_confidence_threshold": YOLO_CONF,
            "yolo_imgsz": YOLO_IMGSZ,
            "yolo_device": "cpu",
            "gcn_architecture": "GCN/STGCNPP/model_multi_person.py",
            "gcn_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "gcn_checkpoint_bytes": CHECKPOINT.stat().st_size,
            "gcn_device": "cpu",
            "gcn_training_metadata": checkpoint_metadata,
        },
        "input_path": {
            "every_decoded_frame_sent_to_yolo": True,
            "original_frame_array_passed_directly": True,
            "external_frame_resize": False,
            "external_image_compression": False,
            "optical_flow": False,
            "ncnn": False,
            "quantization": False,
            "gcn_input": "Raw YOLO COCO-17 pixel keypoints with confidence, grouped by tracked person for the full clip; GCN performs its own Joint-Motion preparation and 100-frame sampling.",
        },
        "yolo": yolo_report,
        "gcn": gcn_report,
        "predictions_by_person": predictions,
        "interpretation": (
            "This is one labeled KTH clip, not a dataset-wide accuracy measurement. "
            "The expected top-1 label is Hand Waving."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
