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

        elapsed = perf_counter() - started
        self.last_fps = 1.0 / elapsed if elapsed > 0 else 0.0

        annotated_frame = None
        if result is not None:
            plotted_frame = result.plot(
                boxes=True,
                labels=True,
                conf=True,
                kpt_line=True,
            )
            encoded_ok, encoded_image = cv2.imencode(
                ".jpg",
                plotted_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            if encoded_ok:
                image_base64 = base64.b64encode(encoded_image).decode("ascii")
                annotated_frame = f"data:image/jpeg;base64,{image_base64}"

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
