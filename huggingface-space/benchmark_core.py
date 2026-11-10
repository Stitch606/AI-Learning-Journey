from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from process.streaming_inference import StreamingInference

VIDEO_URL = "https://raw.githubusercontent.com/krithicswaroopan/Human-Centric-Video-Stabilization/main/test_video1.mp4"


def stats(values):
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values) * 1000, 2),
        "median_ms": round(statistics.median(values) * 1000, 2),
        "p95_ms": round(float(np.percentile(values, 95)) * 1000, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--warmup-frames", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    artifact_dir = Path("artifacts/core-benchmark")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    video_path = artifact_dir / "input.mp4"
    if not video_path.exists():
        request = urllib.request.Request(VIDEO_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            video_path.write_bytes(response.read())
    if video_path.stat().st_size < 100_000:
        raise RuntimeError("Benchmark video download is invalid")

    print({
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
    }, flush=True)

    pipeline = StreamingInference()
    timings = {name: [] for name in ("yolo_track", "stream_manager", "stgcn_predict", "whole_frame")}
    counts = {"frames": 0, "tracked_person_frames": 0, "prediction_calls": 0, "predicted_persons": 0}

    def wrap(target, method_name, bucket, callback=None):
        original = getattr(target, method_name)
        def measured(*call_args, **call_kwargs):
            started = time.perf_counter()
            result = original(*call_args, **call_kwargs)
            timings[bucket].append(time.perf_counter() - started)
            if callback:
                callback(result)
            return result
        setattr(target, method_name, measured)

    def track_count(results):
        result = results[0] if results else None
        if result is not None and result.boxes is not None and result.boxes.id is not None:
            counts["tracked_person_frames"] += len(result.boxes.id)

    def prediction_count(result):
        counts["prediction_calls"] += 1
        counts["predicted_persons"] += len(result.get("persons", []))

    wrap(pipeline.yolo, "track", "yolo_track", track_count)
    wrap(pipeline.manager, "update", "stream_manager")
    wrap(pipeline.mlti, "predict", "stgcn_predict", prediction_count)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the downloaded video")

    try:
        while counts["frames"] < args.max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            if w > 640:
                bgr = cv2.resize(bgr, (640, round(h * 640 / w)), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            pipeline.process_frame(rgb)
            elapsed = time.perf_counter() - started
            if counts["frames"] >= args.warmup_frames:
                timings["whole_frame"].append(elapsed)
            counts["frames"] += 1
            if counts["frames"] % 10 == 0:
                print(f"Processed {counts['frames']} frames", flush=True)
    finally:
        capture.release()

    if counts["frames"] <= args.warmup_frames:
        raise RuntimeError(f"Only {counts['frames']} frames were read from the test video")

    frame_times = timings["whole_frame"]
    mean_frame_s = statistics.fmean(frame_times)
    report = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cpu_count": os.cpu_count(),
        "input_resolution_max_width": 640,
        "frames_processed": counts["frames"],
        "warmup_frames_excluded": args.warmup_frames,
        "tracked_person_frames": counts["tracked_person_frames"],
        "stgcn_prediction_calls": counts["prediction_calls"],
        "persons_predicted": counts["predicted_persons"],
        "overall": {
            "mean_fps": round(1 / mean_frame_s, 2),
            "mean_frame_ms": round(mean_frame_s * 1000, 2),
            "median_frame_ms": round(statistics.median(frame_times) * 1000, 2),
            "p95_frame_ms": round(float(np.percentile(frame_times, 95)) * 1000, 2),
        },
        "stages": {name: stats(values) for name, values in timings.items() if name != "whole_frame"},
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("CORE BENCHMARK REPORT")
    print(json.dumps(report, indent=2))

    if counts["tracked_person_frames"] == 0:
        raise RuntimeError("YOLO did not produce any tracked people in the sample video")
    if counts["prediction_calls"] == 0:
        raise RuntimeError("No ST-GCN++ prediction occurred; sample video did not fill a person buffer")


if __name__ == "__main__":
    main()
