from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .stream_manager import StreamManager


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = BASE_DIR / "weights"

DEFAULT_YOLO_MODEL = "yolo26m-pose.pt"
DEFAULT_STGCN_CHECKPOINT = (
    WEIGHTS_DIR / "stgcnpp_jm_ntu60_xsub_hrnet_finetuned.pth"
)
MAX_PERSONS = 10
KEYPOINT_CONFIDENCE = 0.25

# COCO-17 skeleton connections (zero-based keypoint indices).
SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3),
    (2, 4), (0, 5), (0, 6),
)


def load_models(
    yolo_model=DEFAULT_YOLO_MODEL,
    stgcn_checkpoint=DEFAULT_STGCN_CHECKPOINT,
    device=None,
):
    """Load YOLO Pose and the project's standalone ST-GCN++ model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO(str(yolo_model))

    model_file = MODELS_DIR / "ST_GCN++.py"
    spec = importlib.util.spec_from_file_location("stgcnpp_architecture", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load ST-GCN++ architecture from {model_file}")

    architecture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(architecture)

    mlti = architecture.STGCNPPMultiPerson(
        checkpoint_path=str(stgcn_checkpoint),
        device=device,
        max_persons=MAX_PERSONS,
    )

    return yolo, mlti


class StreamingInference:
    """Persistent per-frame inference state for the Gradio frame endpoint."""

    def __init__(self, yolo=None, mlti=None):
        if yolo is None or mlti is None:
            loaded_yolo, loaded_mlti = load_models()
            yolo = yolo or loaded_yolo
            mlti = mlti or loaded_mlti

        self.yolo = yolo
        self.mlti = mlti
        self.manager = StreamManager()
        self.predictions = {}
        self.last_frame_at = None
        self.last_fps = 0.0

    def reset(self):
        self.manager.reset()
        self.predictions.clear()
        self.last_frame_at = None

    def _render_frame(self, bgr_frame, result, detected_ids):
        """Draw tracked boxes, COCO-17 pose skeletons, IDs, and action labels."""
        canvas = bgr_frame.copy()
        if result is None or result.boxes is None:
            return canvas

        boxes = result.boxes.xyxy
        if boxes is None or len(boxes) == 0:
            return canvas

        boxes = boxes.detach().cpu().numpy()
        keypoints = None
        if result.keypoints is not None and result.keypoints.data is not None:
            keypoints = result.keypoints.data.detach().cpu().numpy()

        ids = []
        if result.boxes.id is not None:
            ids = result.boxes.id.detach().cpu().numpy().astype(np.int64).tolist()
        else:
            ids = [None] * len(boxes)

        height, width = canvas.shape[:2]
        line_width = max(2, round(min(width, height) / 360))
        radius = max(3, line_width + 1)
        palette = (
            (80, 220, 120),
            (255, 170, 60),
            (220, 100, 240),
            (60, 210, 240),
            (240, 110, 90),
        )

        for index, box in enumerate(boxes):
            track_id = ids[index] if index < len(ids) else None
            color = palette[(int(track_id) if track_id is not None else index) % len(palette)]
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)

            action = "Person"
            if track_id is not None:
                prediction = self.predictions.get(int(track_id))
                action = prediction["action"] if prediction else "Collecting frames"
                label = f"ID {int(track_id)} | {action}"
            else:
                label = action

            font_scale = max(0.45, min(width, height) / 1000)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
            )
            label_top = max(0, y1 - text_height - baseline - 8)
            cv2.rectangle(
                canvas,
                (x1, label_top),
                (min(width - 1, x1 + text_width + 8), y1),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 4, max(text_height, y1 - baseline - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (20, 24, 30),
                line_width,
                cv2.LINE_AA,
            )

            if keypoints is None or index >= len(keypoints):
                continue

            person_points = keypoints[index]
            visible = {}
            for point_index, point in enumerate(person_points):
                px, py = int(round(float(point[0]))), int(round(float(point[1])))
                confidence = float(point[2]) if len(point) > 2 else 1.0
                if confidence >= KEYPOINT_CONFIDENCE and 0 <= px < width and 0 <= py < height:
                    visible[point_index] = (px, py)

            for start, end in SKELETON:
                if start in visible and end in visible:
                    cv2.line(
                        canvas,
                        visible[start],
                        visible[end],
                        color,
                        line_width,
                        cv2.LINE_AA,
                    )

            for point in visible.values():
                cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, point, max(1, radius // 2), (245, 245, 245), -1, cv2.LINE_AA)

        return canvas

    @staticmethod
    def _encode_annotated_frame(bgr_frame):
        success, encoded = cv2.imencode(
            ".jpg",
            bgr_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 78],
        )
        if not success:
            return None
        image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
        return "data:image/jpeg;base64," + image_data

    def process_frame(self, rgb_frame):
        started = perf_counter()
        now = monotonic()

        if (
            self.last_frame_at is not None
            and now - self.last_frame_at > self.manager.disappear_timeout
        ):
            self.reset()

        self.last_frame_at = now

        frame = np.asarray(rgb_frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected an RGB frame with shape [height, width, 3]")

        height, width = frame.shape[:2]
        bgr_frame = np.ascontiguousarray(frame[:, :, ::-1])
        results = self.yolo.track(
            bgr_frame,
            persist=True,
            verbose=False,
        )

        detected = []
        ready = []
        result = results[0] if results else None

        if result is not None and result.keypoints is not None:
            keypoints_tensor = result.keypoints.data

            if result.boxes is not None and result.boxes.id is not None:
                keypoints = keypoints_tensor.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(np.int64)

                ready = self.manager.update(
                    keypoints=keypoints,
                    track_ids=track_ids,
                )

                for track_id in track_ids:
                    detected.append(int(track_id))
            else:
                self.manager.update(
                    keypoints=np.empty((0, 17, 3), dtype=np.float32),
                    track_ids=np.empty((0,), dtype=np.int64),
                )
        else:
            self.manager.update(
                keypoints=np.empty((0, 17, 3), dtype=np.float32),
                track_ids=np.empty((0,), dtype=np.int64),
            )

        if ready:
            person_ids = [person["track_id"] for person in ready]
            ready_keypoints = np.stack(
                [person["keypoints"] for person in ready],
                axis=0,
            )

            prediction = self.mlti.predict(
                ready_keypoints,
                topk=5,
                person_ids=person_ids,
                image_size=(width, height),
            )

            for person in prediction["persons"]:
                self.predictions[int(person["person_id"])] = {
                    "action": person["action"],
                    "confidence": person["confidence"],
                }

        active_tracks = set(self.manager.get_active_tracks())
        self.predictions = {
            track_id: prediction
            for track_id, prediction in self.predictions.items()
            if track_id in active_tracks
        }

        people = []
        for track_id in detected:
            prediction = self.predictions.get(track_id)
            people.append({
                "id": track_id,
                "person": f"Person {track_id}",
                "action": prediction["action"] if prediction else "Collecting frames",
                "confidence": prediction["confidence"] if prediction else 0.0,
            })

        rendered_frame = self._render_frame(bgr_frame, result, detected)
        annotated_frame = self._encode_annotated_frame(rendered_frame)

        elapsed = perf_counter() - started
        self.last_fps = 1.0 / elapsed if elapsed > 0 else 0.0

        return {
            "people": people,
            "fps": self.last_fps,
            "model_connected": True,
            "modelStatus": "YOLO Pose + ST-GCN++ connected",
            "message": "Collecting 100 frames per person before the first action prediction.",
            "frame_shape": list(frame.shape),
            "frame_dtype": str(frame.dtype),
            "annotated_frame": annotated_frame,
        }


_streaming_inference = None
_streaming_lock = Lock()


def process_stream_frame(rgb_frame):
    """Process one RGB frame while preserving tracker and person buffers."""
    global _streaming_inference

    with _streaming_lock:
        if _streaming_inference is None:
            _streaming_inference = StreamingInference()
        return _streaming_inference.process_frame(rgb_frame)


def reset_streaming_state():
    """Clear tracker and temporal buffers before starting a fresh stream."""
    global _streaming_inference

    with _streaming_lock:
        if _streaming_inference is not None:
            _streaming_inference.reset()
