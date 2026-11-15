from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from gpu import gpu
from process.streaming_inference import process_stream_frame, reset_streaming_state


def get_video_progress(progress_path: str) -> dict:
    """Read progress written by the isolated ZeroGPU worker."""
    try:
        return json.loads(Path(progress_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _set_video_progress(progress_path: str, **values) -> None:
    """Atomically publish progress so the web process can poll it."""
    path = Path(progress_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values), encoding="utf-8")
    os.replace(temporary, path)


@gpu(duration=120)
def analyze_video_file(input_path: str, progress_path: str) -> dict:
    """Analyze an uploaded video on the GPU worker and return portable MP4 bytes.

    ZeroGPU executes this function in an isolated worker process. A filesystem
    path created by that worker must not be treated as the web process' output.
    Returning the encoded bytes makes the hand-off explicit and reliable.
    """
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if not source_fps or source_fps < 1 or source_fps > 240:
        source_fps = 30.0
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = source_frames / source_fps if source_frames else 0.0

    sample_interval = max(1, round(source_fps / 6.0))
    output_fps = source_fps / sample_interval
    processed_frames = 0
    frame_index = 0
    people_summary = {}
    expected_samples = (source_frames + sample_interval - 1) // sample_interval if source_frames else 0

    output_fd, worker_output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(output_fd)
    os.unlink(worker_output_path)
    writer = None
    codec_used = None
    _set_video_progress(progress_path, stage="processing", progress=0.0, processed_frames=0, total_frames=expected_samples)

    reset_streaming_state()
    try:
        while True:
            ok, bgr_frame = capture.read()
            if not ok:
                break

            if frame_index % sample_interval == 0:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                result = process_stream_frame(rgb_frame)

                for person in result.get("people", []):
                    person_id = int(person["id"])
                    people_summary[person_id] = {
                        "id": person_id,
                        "person": person.get("person", f"Person {person_id}"),
                        "action": person.get("action", "Unknown"),
                        "confidence": float(person.get("confidence", 0.0)),
                        "time": f"{frame_index / source_fps:.1f}s",
                    }

                encoded = result.get("annotated_frame")
                if encoded and encoded.startswith("data:image/"):
                    image_bytes = base64.b64decode(encoded.split(",", 1)[1])
                    annotated = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if annotated is None:
                        raise RuntimeError("Could not decode the annotated frame.")
                else:
                    annotated = bgr_frame

                if writer is None:
                    height, width = annotated.shape[:2]
                    for codec in ("avc1", "mp4v"):
                        candidate = cv2.VideoWriter(
                            worker_output_path,
                            cv2.VideoWriter_fourcc(*codec),
                            output_fps,
                            (width, height),
                        )
                        if candidate.isOpened():
                            writer = candidate
                            codec_used = codec
                            break
                        candidate.release()
                    if writer is None:
                        raise RuntimeError("No MP4 video encoder is available on this Space.")

                writer.write(annotated)
                processed_frames += 1
                if expected_samples:
                    _set_video_progress(
                        progress_path,
                        stage="processing",
                        progress=min(99.0, processed_frames * 100.0 / expected_samples),
                        processed_frames=processed_frames,
                        total_frames=expected_samples,
                    )

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        reset_streaming_state()

    try:
        output_file = Path(worker_output_path)
        if processed_frames == 0 or not output_file.is_file() or output_file.stat().st_size == 0:
            raise ValueError("No readable video frames were found or the MP4 encoder produced an empty file.")

        video_bytes = output_file.read_bytes()
        if not video_bytes:
            raise ValueError("The analyzed video encoder returned an empty result.")

        _set_video_progress(
            progress_path,
            stage="completed",
            progress=100.0,
            processed_frames=processed_frames,
            total_frames=expected_samples,
        )
        return {
            "video_bytes": video_bytes,
            "frames_analyzed": processed_frames,
            "duration_seconds": round(duration_seconds, 2),
            "output_fps": round(output_fps, 3),
            "codec": codec_used or "unknown",
            "people": list(people_summary.values()),
        }
    finally:
        try:
            os.remove(worker_output_path)
        except FileNotFoundError:
            pass
