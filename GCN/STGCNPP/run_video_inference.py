"""Run standalone ST-GCN++ JM on a real video URL using YOLO COCO-17 pose extraction.

This is an end-to-end inference smoke test, not an accuracy benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# run_video_inference.py is <repo>/GCN/STGCNPP/run_video_inference.py.
# parents[2] is the repository root; parents[3] points one directory too high.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from GCN.STGCNPP import STGCNPP


def extract_pose(video_path: Path, pose_model: YOLO, max_frames: int = 180):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open downloaded video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / 15.0)))

    frames = []
    frame_index = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride == 0:
            result = pose_model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
            chosen = np.zeros((17, 3), dtype=np.float32)
            if result.keypoints is not None and result.keypoints.data is not None:
                candidates = result.keypoints.data.detach().cpu().numpy()
                if len(candidates):
                    scores = candidates[:, :, 2].mean(axis=1)
                    chosen = candidates[int(np.argmax(scores))].astype(np.float32)
            frames.append(chosen)
        frame_index += 1
    cap.release()

    if not frames:
        raise RuntimeError("No frames were decoded from the video")
    return np.stack(frames), (width, height), fps, frame_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--name", default="sample")
    parser.add_argument("--checkpoint", default="GCN/checkpoints/stgcnpp_jm_ntu60_xsub_hrnet.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=180)
    args = parser.parse_args()

    out_dir = ROOT / "artifacts" / "stgcnpp-video-test"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / (args.name + ".mp4")
    print("Downloading video:", args.video_url, flush=True)
    urlretrieve(args.video_url, video_path)
    if video_path.stat().st_size < 10000:
        raise RuntimeError("Downloaded video is unexpectedly small")

    print("Loading YOLOv8n-pose COCO-17 extractor", flush=True)
    pose_model = YOLO("yolov8n-pose.pt")
    keypoints, image_size, fps, decoded = extract_pose(
        video_path, pose_model, max_frames=args.max_frames)
    detected = int((keypoints[:, :, 2].mean(axis=1) > 0.01).sum())
    print("Frames processed:", len(keypoints), "of", decoded)
    print("Frames with a detected person:", detected)

    print("Loading standalone ST-GCN++ and official JM checkpoint", flush=True)
    model = STGCNPP(
        checkpoint_path=args.checkpoint,
        num_classes=60,
        device=args.device,
        strict=True,
    )
    probabilities = model.predict(keypoints, image_size=image_size)[0]
    values, indices = torch.topk(probabilities, k=5)
    labels_path = ROOT / "GCN" / "STGCNPP" / "ntu60_labels.txt"
    labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
    rows = [
        {"rank": rank + 1, "class_id": int(class_id),
         "label": labels[int(class_id)], "probability": float(prob)}
        for rank, (class_id, prob) in enumerate(zip(indices.cpu(), values.cpu()))
    ]
    report = {
        "video_name": args.name,
        "video_url": args.video_url,
        "video_size": list(image_size),
        "source_fps": fps,
        "pose_frames": int(len(keypoints)),
        "frames_with_person": detected,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "top5": rows,
        "note": "Inference smoke test only; top-1 output is not a verified ground-truth accuracy score."
    }
    report_path = out_dir / (args.name + ".json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if detected < 10:
        raise RuntimeError("Too few person detections for a meaningful video inference run")


if __name__ == "__main__":
    main()
