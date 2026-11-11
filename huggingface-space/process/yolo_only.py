from __future__ import annotations

import base64
from threading import Lock
from time import perf_counter

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_YOLO_MODEL = "yolo26m-pose.pt"
KEYPOINT_CONFIDENCE = 0.25
SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3),
    (2, 4), (0, 5), (0, 6),
)


class YOLOOnlyInference:
    """YOLO Pose tracking and drawing, with no StreamManager or action model."""

    def __init__(self, model_name=DEFAULT_YOLO_MODEL):
        self.model = YOLO(model_name)

    def reset(self):
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None:
            tracker = getattr(predictor, "trackers", None)
            if tracker:
                for item in tracker:
                    reset = getattr(item, "reset", None)
                    if callable(reset):
                        reset()

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

        if result.boxes.id is not None:
            ids = result.boxes.id.detach().cpu().numpy().astype(np.int64).tolist()
        else:
            ids = [None] * len(boxes)

        height, width = canvas.shape[:2]
        thickness = max(2, round(min(width, height) / 360))
        radius = max(3, thickness + 1)
        palette = (
            (80, 220, 120), (255, 170, 60), (220, 100, 240),
            (60, 210, 240), (240, 110, 90),
        )
        people = []

        for index, box in enumerate(boxes):
            track_id = ids[index] if index < len(ids) else None
            color = palette[(int(track_id) if track_id is not None else index) % len(palette)]
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            label = f"ID {track_id}" if track_id is not None else "Person"
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
                "id": int(track_id) if track_id is not None else index + 1,
                "person": f"Person {track_id}" if track_id is not None else f"Person {index + 1}",
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

    def process_frame(self, rgb_frame):
        started = perf_counter()
        frame = np.asarray(rgb_frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected an RGB frame with shape [height, width, 3]")

        bgr = np.ascontiguousarray(frame[:, :, ::-1])
        results = self.model.track(bgr, persist=True, verbose=False)
        result = results[0] if results else None
        rendered, people = self._render(bgr, result)
        annotated_frame = self._encode(rendered)

        elapsed_ms = (perf_counter() - started) * 1000
        return {
            "people": people,
            "fps": 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
            "model_connected": True,
            "modelStatus": "YOLO Pose only · StreamManager and ST-GCN++ bypassed",
            "message": "YOLO-only isolation test: no action recognition is running.",
            "frame_shape": list(frame.shape),
            "frame_dtype": str(frame.dtype),
            "annotated_frame": annotated_frame,
            "yolo_only_timing": {"process_ms": round(elapsed_ms, 2)},
        }


_yolo_only = None
_yolo_only_lock = Lock()


def process_yolo_only_frame(rgb_frame):
    global _yolo_only
    with _yolo_only_lock:
        if _yolo_only is None:
            _yolo_only = YOLOOnlyInference()
        return _yolo_only.process_frame(rgb_frame)


def reset_yolo_only_state():
    global _yolo_only
    with _yolo_only_lock:
        if _yolo_only is not None:
            _yolo_only.reset()
