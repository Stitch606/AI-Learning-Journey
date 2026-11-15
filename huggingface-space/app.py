from __future__ import annotations

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import os
import tempfile
import base64
import json
from struct import unpack_from
from time import perf_counter
from threading import Lock
from uuid import uuid4

import gradio as gr
import numpy as np
import cv2
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from gpu import gpu, gpu_ready
from process.streaming_inference import process_stream_frame, reset_streaming_state
from process.yolo_only import process_yolo_frame
from process.video_analysis import analyze_video_file, get_video_progress


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


_VIDEO_JOBS = {}
_VIDEO_JOBS_LOCK = Lock()
_VIDEO_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_VIDEO_RESULT_TTL_SECONDS = 30 * 60


def _cleanup_expired_video_jobs() -> None:
    """Remove completed/failed demo jobs after their display window expires."""
    now = datetime.now(timezone.utc).timestamp()
    expired = []
    with _VIDEO_JOBS_LOCK:
        for job_id, job in list(_VIDEO_JOBS.items()):
            created_at = float(job.get("created_at", now))
            if job.get("status") in ("completed", "failed") and now - created_at > _VIDEO_RESULT_TTL_SECONDS:
                expired.append((job_id, job.get("output_path")))
                _VIDEO_JOBS.pop(job_id, None)
    for _, output_path in expired:
        if output_path:
            try:
                os.remove(output_path)
            except FileNotFoundError:
                pass



def _run_video_job(job_id: str, input_path: str, output_path: str, progress_path: str) -> None:
    with _VIDEO_JOBS_LOCK:
        _VIDEO_JOBS[job_id].update(status="processing", stage="processing")
    try:
        result = analyze_video_file(input_path, progress_path)
        video_bytes = result.pop("video_bytes", None)
        if not isinstance(video_bytes, bytes) or not video_bytes:
            raise RuntimeError("The GPU worker completed without returning an MP4 video.")
        if len(video_bytes) > 256 * 1024 * 1024:
            raise RuntimeError("The analyzed video exceeds the 256 MB result limit.")
        with open(output_path, "wb") as destination:
            destination.write(video_bytes)
            destination.flush()
            os.fsync(destination.fileno())
        if not os.path.isfile(output_path) or os.path.getsize(output_path) != len(video_bytes):
            raise RuntimeError("The web server could not persist the analyzed video returned by the GPU worker.")
        with _VIDEO_JOBS_LOCK:
            _VIDEO_JOBS[job_id].update(
                status="completed",
                stage="completed",
                progress=100.0,
                result=result,
                output_ready=True,
                output_size=len(video_bytes),
            )
    except Exception as error:
        with _VIDEO_JOBS_LOCK:
            _VIDEO_JOBS[job_id].update(
                status="failed",
                stage="failed",
                error=f"{type(error).__name__}: {error}",
            )
        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass
    finally:
        for path in (input_path, progress_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


@app.post("/api/analyze-video", status_code=202)
async def analyze_video_upload(file: UploadFile = File(...)):
    """Accept an upload quickly, then analyze it in a background worker."""
    _cleanup_expired_video_jobs()
    filename = (file.filename or "video").lower()
    if not filename.endswith((".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")):
        raise HTTPException(status_code=415, detail="Upload an MP4, MOV, M4V, AVI, MKV, or WebM video.")

    input_fd, input_path = tempfile.mkstemp(suffix=Path(filename).suffix or ".mp4")
    os.close(input_fd)
    output_fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(output_fd)
    os.unlink(output_path)
    progress_fd, progress_path = tempfile.mkstemp(suffix=".json")
    os.close(progress_fd)

    try:
        total_bytes = 0
        with open(input_path, "wb") as destination:
            while True:
                part = await file.read(1024 * 1024)
                if not part:
                    break
                total_bytes += len(part)
                if total_bytes > 512 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Video is larger than the 512 MB demo limit.")
                destination.write(part)

        job_id = uuid4().hex
        with _VIDEO_JOBS_LOCK:
            _VIDEO_JOBS[job_id] = {
                "status": "queued",
                "stage": "queued",
                "progress": 0.0,
                "filename": Path(filename).name,
                "output_path": output_path,
                "progress_path": progress_path,
                "created_at": datetime.now(timezone.utc).timestamp(),
            }
        _VIDEO_EXECUTOR.submit(_run_video_job, job_id, input_path, output_path, progress_path)
        return {"job_id": job_id, "status": "queued"}
    except HTTPException:
        for path in (input_path, output_path, progress_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise
    except Exception as error:
        for path in (input_path, output_path, progress_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise HTTPException(status_code=500, detail=f"Could not accept video: {error}") from error
    finally:
        await file.close()


@app.get("/api/analyze-video/{job_id}")
async def get_video_analysis_status(job_id: str):
    with _VIDEO_JOBS_LOCK:
        job = dict(_VIDEO_JOBS.get(job_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail="Video analysis job was not found.")
    if job.get("status") == "processing" and job.get("progress_path"):
        job.update(get_video_progress(job["progress_path"]))
    job.pop("output_path", None)
    job.pop("progress_path", None)
    result = job.pop("result", None)
    if result:
        job["result"] = {
            "frames_analyzed": result["frames_analyzed"],
            "duration_seconds": result["duration_seconds"],
            "output_fps": result["output_fps"],
            "codec": result["codec"],
            "people": result["people"],
        }
    return job


@app.get("/api/analyze-video/{job_id}/result")
async def get_video_analysis_result(job_id: str):
    """Serve the analyzed MP4 directly to the browser's native video player."""
    with _VIDEO_JOBS_LOCK:
        job = _VIDEO_JOBS.get(job_id)
        if not job or job.get("status") != "completed":
            raise HTTPException(status_code=409, detail="The analyzed video is not ready yet.")
        output_path = job.get("output_path")
        result = job.get("result")

    if not output_path or not os.path.isfile(output_path):
        raise HTTPException(status_code=500, detail="Analysis finished, but the output video file is missing on the Space server.")
    if os.path.getsize(output_path) <= 0:
        raise HTTPException(status_code=500, detail="The analyzed video file is empty.")

    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-store",
        "Content-Disposition": 'inline; filename="analyzed-video.mp4"',
        "X-Frames-Analyzed": str(result["frames_analyzed"]),
        "X-Video-Duration": str(result["duration_seconds"]),
        "X-Analysis-FPS": str(result["output_fps"]),
        "X-Video-Codec": result["codec"],
        "X-Action-Summary": base64.b64encode(json.dumps(result["people"]).encode("utf-8")).decode("ascii"),
    }
    # Keep the result file available for metadata, playback, and seek requests.
    return FileResponse(path=output_path, media_type="video/mp4", headers=headers)


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
    """Decode the legacy HAB1 frame container used by the action-recognition endpoint."""
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

    return frames, (perf_counter() - decode_started) * 1000


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



@app.api(name="receive_yolo_frame", queue=False, description="Run YOLO on one camera frame.")
@gpu
def receive_yolo_frame(frame_id: int, frame: dict) -> dict:
    if frame is None:
        return {"status": "error", "message": "No frame received."}

    rgb_frame = np.asarray(load_frame(frame), dtype=np.uint8)
    return {
        "status": "ok",
        "frame_id": frame_id,
        **process_yolo_frame(rgb_frame),
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
        "message": "Use the receive_batch endpoint to stream frames.",
        "received_video": bool(video_path),
    }


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
