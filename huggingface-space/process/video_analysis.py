from __future__ import annotations

import base64
from time import perf_counter

import cv2
import numpy as np

from gpu import gpu
from process.streaming_inference import process_stream_frame


@gpu(duration=120)
def process_video_chunk(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Run an ordered chunk through the existing YOLO + ST-GCN++ pipeline."""
    rendered = []
    for rgb_frame in frames:
        result = process_stream_frame(rgb_frame)
        encoded = result.get("annotated_frame")
        if encoded and encoded.startswith("data:image/"):
            image_bytes = base64.b64decode(encoded.split(",", 1)[1])
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("The annotated frame could not be decoded.")
            rendered.append(image)
        else:
            rendered.append(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    return rendered
