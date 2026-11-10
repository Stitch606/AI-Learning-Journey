from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from .stream_manager import StreamManager
except ImportError:
    from stream_manager import StreamManager


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = BASE_DIR / "weights"

DEFAULT_YOLO_MODEL = "yolo26m-pose.pt"
DEFAULT_STGCN_CHECKPOINT = (
    WEIGHTS_DIR / "stgcnpp_jm_ntu60_xsub_hrnet_finetuned.pth"
)
MAX_PERSONS = 10


class capture:
    def __init__(self, video):
        self.video = video

        self.cap = cv2.VideoCapture(self.video)
        if not self.cap.isOpened():
            print("the video is broken")

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()


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


def run_streaming_inference(
    video_obj,
    yolo=None,
    mlti=None,
    yolo_model=DEFAULT_YOLO_MODEL,
    stgcn_checkpoint=DEFAULT_STGCN_CHECKPOINT,
):
    """Run video capture -> YOLO tracking/pose -> per-person ST-GCN++ prediction."""
    if isinstance(video_obj, (str, Path, int)):
        video_obj = capture(str(video_obj) if isinstance(video_obj, Path) else video_obj)

    owns_capture = isinstance(video_obj, capture)

    if yolo is None or mlti is None:
        loaded_yolo, loaded_mlti = load_models(
            yolo_model=yolo_model,
            stgcn_checkpoint=stgcn_checkpoint,
        )
        if yolo is None:
            yolo = loaded_yolo
        if mlti is None:
            mlti = loaded_mlti

    manager = StreamManager()

    try:
        while True:
            ret, frame = video_obj.read()
            if not ret:
                break

            results = yolo.track(
                frame,
                persist=True,
                verbose=False,
            )

            if (
                len(results) > 0
                and results[0].boxes is not None
                and results[0].boxes.id is not None
                and results[0].keypoints is not None
            ):
                result = results[0]

                keypoints = result.keypoints.data.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy()

                ready = manager.update(
                    keypoints=keypoints,
                    track_ids=track_ids,
                )

                if ready:
                    person_ids = [person["track_id"] for person in ready]

                    ready_keypoints = np.stack(
                        [person["keypoints"] for person in ready],
                        axis=0,
                    )

                    prediction = mlti.predict(
                        ready_keypoints,
                        topk=5,
                        person_ids=person_ids,
                        image_size=(frame.shape[1], frame.shape[0]),
                    )

                    print(prediction)

            else:
                manager.update(
                    keypoints=np.empty((0, 17, 3), dtype=np.float32),
                    track_ids=np.empty((0,), dtype=np.int64),
                )
    finally:
        if owns_capture:
            video_obj.release()
