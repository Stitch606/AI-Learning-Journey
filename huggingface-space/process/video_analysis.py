from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from gpu import gpu
from process.streaming_inference import process_stream_frame, reset_streaming_state


@gpu(duration=120)
def analyze_video_file(input_path: str, output_path: str) -> dict:
    """Analyze one uploaded video in a single GPU reservation and write an annotated MP4."""
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
    writer = None
    codec_used = None
    processed_frames = 0
    frame_index = 0

    reset_streaming_state()
    try:
        while True:
            ok, bgr_frame = capture.read()
            if not ok:
                break
            if frame_index % sample_interval == 0:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                result = process_stream_frame(rgb_frame)
                encoded = result.get("annotated_frame")
                if encoded and encoded.startswith("data:image/"):
                    image_bytes = base64.b64decode(encoded.split(",", 1)[1])
                    annotated = cv2.imdecode(
                        np.frombuffer(image_bytes, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if annotated is None:
                        raise RuntimeError("Could not decode the annotated frame.")
                else:
                    annotated = bgr_frame

                if writer is None:
                    height, width = annotated.shape[:2]
                    for codec in ("avc1", "mp4v"):
                        candidate = cv2.VideoWriter(
                            output_path,
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
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        reset_streaming_state()

    if processed_frames == 0 or not Path(output_path).is_file() or Path(output_path).stat().st_size == 0:
        raise ValueError("No readable video frames were found.")

    return {
        "frames_analyzed": processed_frames,
        "duration_seconds": round(duration_seconds, 2),
        "output_fps": round(output_fps, 3),
        "codec": codec_used or "unknown",
    }
