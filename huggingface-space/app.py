from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path

import gradio as gr
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from gpu import gpu, gpu_ready
from process.streaming_inference import process_stream_frame, reset_streaming_state


BASE_DIR = Path(__file__).resolve().parent

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
    Gradio Server API receives uploaded files as FileData dictionaries.
    Resolve the temporary file path and open it as a PIL image.
    """
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, dict):
        frame_path = frame.get("path")
        if frame_path:
            with Image.open(frame_path) as image:
                return image.convert("RGB")

    raise ValueError("The Gradio frame payload did not contain a readable image path.")


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
        "message": "Use the receive_frame endpoint to stream frames.",
        "received_video": bool(video_path),
    }


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
