from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from gpu import gpu
from process.streaming_inference import (
    process_flow_stream_frame,
    process_stream_frame,
    reset_streaming_state,
)


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


def _start_h264_encoder(output_path: str, width: int, height: int, fps: float):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=20,
        check=False,
    )
    if probe.returncode != 0 or "libx264" not in probe.stdout:
        raise RuntimeError("The bundled FFmpeg does not provide the H.264 libx264 encoder.")

    error_log = tempfile.TemporaryFile()
    process = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            output_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=error_log,
        bufsize=1024 * 1024,
    )
    return process, error_log


def _finish_h264_encoder(process, error_log) -> None:
    if process is None:
        if error_log is not None:
            error_log.close()
        return

    try:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    except (BrokenPipeError, OSError):
        pass

    try:
        return_code = process.wait(timeout=90)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait()
        raise RuntimeError("H.264 video encoding did not finish within 90 seconds.") from error

    try:
        error_log.seek(0)
        error_message = error_log.read().decode("utf-8", errors="replace").strip()
    finally:
        error_log.close()

    if return_code != 0:
        raise RuntimeError("H.264 video encoding failed: " + (error_message[-1200:] or f"FFmpeg exit code {return_code}"))


@gpu(duration=120)
def analyze_video_file(input_path: str, progress_path: str) -> dict:
    """Analyze a video on the GPU worker and return browser-compatible H.264 MP4 bytes.

    ZeroGPU executes this function in an isolated worker process. The encoded
    video is returned as bytes so the web process can persist and serve it.
    """
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if not source_fps or source_fps < 1 or source_fps > 240:
        source_fps = 30.0
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = source_frames / source_fps if source_frames else 0.0

    sample_interval = max(1, round(source_fps / 10.0))
    output_fps = source_fps
    processed_frames = 0
    yolo_frames = 0
    frame_index = 0
    people_summary = {}
    expected_samples = source_frames

    output_fd, worker_output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(output_fd)
    os.unlink(worker_output_path)
    encoder = None
    error_log = None
    codec_used = "h264"
    _set_video_progress(progress_path, stage="processing", progress=0.0, processed_frames=0, total_frames=expected_samples)

    reset_streaming_state()
    try:
        while True:
            ok, bgr_frame = capture.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            if frame_index % sample_interval == 0:
                result = process_stream_frame(rgb_frame)
                yolo_frames += 1
            else:
                result = process_flow_stream_frame(rgb_frame)

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
                annotated = cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if annotated is None:
                    raise RuntimeError("Could not decode the annotated frame.")
            else:
                annotated = bgr_frame

            if encoder is None:
                height, width = annotated.shape[:2]
                encoder, error_log = _start_h264_encoder(
                    worker_output_path,
                    width,
                    height,
                    output_fps,
                )

            try:
                encoder.stdin.write(np.ascontiguousarray(annotated).tobytes())
            except (BrokenPipeError, OSError) as error:
                raise RuntimeError("The H.264 encoder stopped while receiving video frames.") from error

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
        try:
            _finish_h264_encoder(encoder, error_log)
        finally:
            reset_streaming_state()

    try:
        output_file = Path(worker_output_path)
        if processed_frames == 0 or not output_file.is_file() or output_file.stat().st_size == 0:
            raise ValueError("No readable video frames were found or the H.264 encoder produced an empty file.")

        validation = cv2.VideoCapture(str(output_file))
        valid_video = validation.isOpened() and int(validation.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
        validation.release()
        if not valid_video:
            raise ValueError("The H.264 output could not be decoded after encoding.")

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
            "yolo_frames": yolo_frames,
            "duration_seconds": round(duration_seconds, 2),
            "output_fps": round(output_fps, 3),
            "codec": codec_used,
            "people": list(people_summary.values()),
        }
    finally:
        try:
            os.remove(worker_output_path)
        except FileNotFoundError:
            pass
