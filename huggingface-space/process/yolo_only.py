from __future__ import annotations

import base64
from multiprocessing import Event, Queue
from pathlib import Path
from queue import Empty
from threading import Lock
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.loaders import LoadPilAndNumpy, SourceTypes


DEFAULT_YOLO_MODEL = "yolo26m-pose.pt"
IMAGE_SIZE = 640
KEYPOINT_CONFIDENCE = 0.25
SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3),
    (2, 4), (0, 5), (0, 6),
)


class QueueFrameLoader(LoadPilAndNumpy):
    """A persistent Ultralytics source fed by frames arriving from Gradio."""

    def __init__(self, frames: Queue, stopped: Event):
        self.frames = frames
        self.stopped = stopped
        self.paths = []
        self.im0 = []
        self.mode = "stream"
        self.bs = 1
        self.count = 0
        self.current_frame_id = -1
        self.source_type = SourceTypes(
            stream=True, screenshot=False, from_img=False, tensor=False
        )

    def __iter__(self):
        return self

    def __len__(self):
        return 1

    def __next__(self):
        while not self.stopped.is_set():
            try:
                item = self.frames.get(timeout=0.25)
            except Empty:
                continue

            if item is None:
                raise StopIteration

            frame_id, rgb_frame = item
            frame = np.asarray(rgb_frame, dtype=np.uint8)
            if frame.ndim != 3 or frame.shape[2] != 3:
                continue

            self.current_frame_id = int(frame_id)
            bgr = np.ascontiguousarray(frame[:, :, ::-1])
            self.paths = [f"browser-frame-{self.current_frame_id}.jpg"]
            self.im0 = [bgr]
            self.bs = 1
            self.count += 1
            return self.paths, self.im0, [""]

        raise StopIteration


class YOLOOnlyInference:
    """YOLO Pose inference over one persistent, queue-fed TensorRT input stream."""

    def __init__(self, model_name=DEFAULT_YOLO_MODEL):
        pytorch_model = YOLO(model_name)
        checkpoint_path = Path(pytorch_model.ckpt_path)
        engine_path = checkpoint_path.with_suffix(".engine")

        if engine_path.is_file():
            try:
                self.model = YOLO(str(engine_path))
                print(f"[YOLO] Loaded cached TensorRT engine: {engine_path}")
                return
            except Exception as error:
                print(f"[YOLO] Cached engine could not be loaded; rebuilding it: {error}")
                engine_path.unlink(missing_ok=True)

        print("[YOLO] Building TensorRT FP16 engine...")
        exported_path = pytorch_model.export(
            format="engine",
            quantize=16,
            imgsz=IMAGE_SIZE,
            batch=1,
            verbose=False,
        )
        self.model = YOLO(str(exported_path))
        print(f"[YOLO] TensorRT FP16 engine ready: {exported_path}")

    @staticmethod
    def _render(frame, result):
        canvas = frame.copy()
        if result is None or result.boxes is None:
            return canvas, []

        boxes = result.boxes.xyxy
        if boxes is None or len(boxes) == 0:
            return canvas, []

        boxes = boxes.detach().cpu().numpy()
        keypoints = None
        if result.keypoints is not None and result.keypoints.data is not None:
            keypoints = result.keypoints.data.detach().cpu().numpy()

        height, width = canvas.shape[:2]
        thickness = max(2, round(min(width, height) / 360))
        radius = max(3, thickness + 1)
        palette = (
            (80, 220, 120), (255, 170, 60), (220, 100, 240),
            (60, 210, 240), (240, 110, 90),
        )
        people = []

        for index, box in enumerate(boxes):
            color = palette[index % len(palette)]
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            label = f"Person {index + 1}"
            font_scale = max(0.45, min(width, height) / 1000)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            label_top = max(0, y1 - text_height - baseline - 8)
            cv2.rectangle(
                canvas, (x1, label_top),
                (min(width - 1, x1 + text_width + 8), y1), color, -1
            )
            cv2.putText(
                canvas, label, (x1 + 4, max(text_height, y1 - baseline - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 24, 30),
                thickness, cv2.LINE_AA
            )

            if keypoints is not None and index < len(keypoints):
                visible = {}
                for point_index, point in enumerate(keypoints[index]):
                    px, py = int(round(float(point[0]))), int(round(float(point[1])))
                    confidence = float(point[2]) if len(point) > 2 else 1.0
                    if confidence >= KEYPOINT_CONFIDENCE and 0 <= px < width and 0 <= py < height:
                        visible[point_index] = (px, py)

                for start, end in SKELETON:
                    if start in visible and end in visible:
                        cv2.line(canvas, visible[start], visible[end], color, thickness, cv2.LINE_AA)

                for point in visible.values():
                    cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, point, max(1, radius // 2), (245, 245, 245), -1, cv2.LINE_AA)

            people.append({
                "id": index + 1,
                "person": f"Person {index + 1}",
                "action": "YOLO detection only",
                "confidence": 0.0,
            })

        return canvas, people

    @staticmethod
    def _encode(frame):
        success, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78]
        )
        if not success:
            return None
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        return "data:image/jpeg;base64," + payload

    def process_stream(self, frames: Queue, stopped: Event):
        loader = QueueFrameLoader(frames, stopped)
        previous_result_at = perf_counter()

        for result in self.model.predict(
            source=loader,
            stream=True,
            verbose=False,
        ):
            now = perf_counter()
            elapsed = now - previous_result_at
            previous_result_at = now
            frame_id = loader.current_frame_id
            frame = result.orig_img
            rendered, people = self._render(frame, result)
            yield {
                "status": "ok",
                "frame_id": frame_id,
                "people": people,
                "model_connected": True,
                "modelStatus": "YOLO Pose · TensorRT FP16",
                "message": "TensorRT FP16 YOLO stream is running.",
                "frame_shape": list(frame.shape),
                "frame_dtype": str(frame.dtype),
                "annotated_frame": self._encode(rendered),
                "inference_ms": round(elapsed * 1000, 2),
                "fps": (1.0 / elapsed) if elapsed > 0 else 0.0,
            }


class YOLOStreamSession:
    def __init__(self):
        self.stream_id = uuid4().hex
        self.frames = Queue()
        self.stopped = Event()


_yolo_only = None
_yolo_only_lock = Lock()
_sessions = {}
_sessions_lock = Lock()


def create_yolo_stream():
    session = YOLOStreamSession()
    with _sessions_lock:
        _sessions[session.stream_id] = session
    return session.stream_id


def enqueue_yolo_frame(stream_id, frame_id, rgb_frame):
    with _sessions_lock:
        session = _sessions.get(stream_id)
    if session is None or session.stopped.is_set():
        raise ValueError("YOLO stream is not active.")
    session.frames.put((int(frame_id), np.asarray(rgb_frame, dtype=np.uint8)))
    return True


def stop_yolo_stream(stream_id):
    with _sessions_lock:
        session = _sessions.pop(stream_id, None)
    if session is None:
        return False
    session.stopped.set()
    session.frames.put(None)
    return True


def process_yolo_stream(stream_id):
    global _yolo_only
    with _sessions_lock:
        session = _sessions.get(stream_id)
    if session is None:
        yield {"status": "error", "message": "YOLO stream session not found."}
        return

    try:
        with _yolo_only_lock:
            if _yolo_only is None:
                _yolo_only = YOLOOnlyInference()
            yield from _yolo_only.process_stream(session.frames, session.stopped)
    finally:
        session.stopped.set()
        with _sessions_lock:
            _sessions.pop(stream_id, None)
