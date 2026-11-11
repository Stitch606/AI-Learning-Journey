from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from struct import unpack_from
from time import perf_counter

import gradio as gr
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from gpu import gpu, gpu_ready
from process.streaming_inference import process_stream_frame, reset_streaming_state
from process.yolo_only import process_yolo_only_batch, reset_yolo_only_state


BASE_DIR = Path(__file__).resolve().parent
BATCH_MAGIC = b"HAB1"
MAX_BATCH_FRAMES = 30

# Gradio Server is the application server and API bridge.
# The custom frontend is served as static files; inference stays in process/.
app = gr.Server(title="Human Action Detection")

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {"status": "ok", "frontend": "ready", "model": "connected"}


def process_frame(frame: Image.Image) -> dict:
    """Run inference on an already-normalized RGB PIL frame."""
    frame_np = np.asarray(frame, dtype=np.uint8)
    return process_stream_frame(frame_np)


def load_frame(frame: dict | Image.Image) -> Image.Image:
    """
    Resolve one Gradio FileData dictionary and decode it as RGB.
    """
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, dict):
        frame_path = frame.get("path")
        if frame_path:
            with Image.open(frame_path) as image:
                return image.convert("RGB")

    raise ValueError("The Gradio frame payload did not contain a readable image path.")


def load_frame_batch(batch: dict) -> tuple[list[Image.Image], float]:
    """
    Decode the compact HAB1 container produced by the browser.
    Layout: magic(4), frame_count(uint32 BE), repeated length(uint32 BE) + JPEG bytes.
    """
    if not isinstance(batch, dict) or not batch.get("path"):
        raise ValueError("The frame batch payload did not contain a readable file path.")

    raw = Path(batch["path"]).read_bytes()
    if len(raw) < 8 or raw[:4] != BATCH_MAGIC:
        raise ValueError("Invalid frame batch header.")

    frame_count = unpack_from(">I", raw, 4)[0]
    if frame_count < 1 or frame_count > MAX_BATCH_FRAMES:
        raise ValueError(f"Frame batch must contain between 1 and {MAX_BATCH_FRAMES} frames.")

    decode_started = perf_counter()
    offset = 8
    frames = []

    for index in range(frame_count):
        if offset + 4 > len(raw):
            raise ValueError(f"Frame batch ended before frame {index + 1} length.")
        frame_size = unpack_from(">I", raw, offset)[0]
        offset += 4
        end = offset + frame_size
        if frame_size <= 0 or end > len(raw):
            raise ValueError(f"Frame batch contains an invalid frame {index + 1}.")
        with Image.open(BytesIO(raw[offset:end])) as image:
            frames.append(image.convert("RGB"))
        offset = end

    if offset != len(raw):
        raise ValueError("Frame batch contains unexpected trailing data.")

    decode_ms = (perf_counter() - decode_started) * 1000
    return frames, decode_ms


@app.api(name="receive_frame", description="Receive one camera or video frame for action analysis.")
@gpu
def receive_frame(frame: dict) -> dict:
    if frame is None:
        return {"status": "error", "message": "No frame received."}

    decode_started = perf_counter()
    rgb_frame = load_frame(frame)
    decode_ms = (perf_counter() - decode_started) * 1000
    inference_started = perf_counter()
    result = process_frame(rgb_frame)
    inference_ms = (perf_counter() - inference_started) * 1000
    return {
        "status": "ok",
        "transport_timing": {
            "decode_ms": round(decode_ms, 2),
            "inference_ms": round(inference_ms, 2),
        },
        "width": rgb_frame.width,
        "height": rgb_frame.height,
        "received_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }


@app.api(name="receive_batch", description="Receive up to 30 ordered frames in one request.")
@gpu
def receive_batch(batch: dict) -> dict:
    if batch is None:
        return {"status": "error", "message": "No frame batch received."}

    frames, decode_ms = load_frame_batch(batch)
    inference_started = perf_counter()
    result = None

    # Keep frames ordered: YOLO tracking and per-person temporal buffers are stateful.
    for frame in frames:
        result = process_frame(frame)

    inference_ms = (perf_counter() - inference_started) * 1000
    last_frame = frames[-1]
    return {
        "status": "ok",
        "batch_timing": {
            "frame_count": len(frames),
            "decode_ms": round(decode_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "inference_ms_per_frame": round(inference_ms / len(frames), 2),
        },
        "width": last_frame.width,
        "height": last_frame.height,
        "received_at": datetime.now(timezone.utc).isoformat(),
        **(result or {}),
    }


@app.api(name="receive_yolo_only_batch", description="Run one YOLO Pose prediction call over up to 30 frames; tracking disabled.")
@gpu
def receive_yolo_only_batch(batch: dict) -> dict:
    if batch is None:
        return {"status": "error", "message": "No frame batch received."}

    frames, decode_ms = load_frame_batch(batch)
    inference_started = perf_counter()
    results = process_yolo_only_batch([np.asarray(frame, dtype=np.uint8) for frame in frames])
    inference_ms = (perf_counter() - inference_started) * 1000
    last_frame = frames[-1]
    last_result = results[-1] if results else {}
    returned_frames = [
        {
            "annotated_frame": item.get("annotated_frame"),
            "people": item.get("people", []),
            "fps": item.get("fps", 0.0),
        }
        for item in results
    ]
    return {
        "status": "ok",
        "frames": returned_frames,
        "batch_timing": {
            "frame_count": len(frames),
            "returned_frame_count": len(returned_frames),
            "decode_ms": round(decode_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "inference_ms_per_frame": round(inference_ms / len(frames), 2),
        },
        "width": last_frame.width,
        "height": last_frame.height,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "people": last_result.get("people", []),
        "fps": last_result.get("fps", 0.0),
        "model_connected": True,
        "modelStatus": "YOLO Pose batch detection · tracking disabled",
        "message": "YOLO processed the complete frame batch in one predict call.",
    }

@app.api(name="reset_yolo_only", description="Reset YOLO-only model state.")
@gpu
def reset_yolo_only() -> dict:
    reset_yolo_only_state()
    return {"status": "ok", "message": "YOLO-only tracker reset."}


@app.api(name="reset_stream", description="Reset person tracking and temporal buffers.")
@gpu
def reset_stream() -> dict:
    reset_streaming_state()
    return {"status": "ok", "message": "Stream state reset."}


@app.api(name="analyze_video", description="Legacy video-analysis endpoint.")
@gpu
def analyze_video(video_path: str) -> dict:
    return {
        "status": "model_not_connected",
        "message": "Use the receive_batch endpoint to stream frames.",
        "received_video": bool(video_path),
    }


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
