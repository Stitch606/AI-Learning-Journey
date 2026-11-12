from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_YOLO_MODEL = "yolo26m-pose.pt"
IMAGE_SIZE = 640
KEYPOINT_CONFIDENCE = 0.25
SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3),
    (2, 4), (0, 5), (0, 6),
)

_model = None


def get_yolo_model():
    global _model
    if _model is not None:
        return _model

    pytorch_model = YOLO(DEFAULT_YOLO_MODEL)
    checkpoint_path = Path(pytorch_model.ckpt_path)
    engine_path = checkpoint_path.with_suffix(".engine")

    if engine_path.is_file():
        try:
            _model = YOLO(str(engine_path))
            print(f"[YOLO] Loaded cached TensorRT engine: {engine_path}")
            return _model
        except Exception as error:
            print(f"[YOLO] Cached TensorRT engine could not be loaded: {error}")
            engine_path.unlink(missing_ok=True)

    print("[YOLO] Building TensorRT FP16 engine...")
    engine_path = pytorch_model.export(
        format="engine",
        quantize=16,
        imgsz=IMAGE_SIZE,
        batch=1,
        verbose=False,
    )
    _model = YOLO(str(engine_path))
    print(f"[YOLO] TensorRT FP16 engine ready: {engine_path}")
    return _model


def draw_yolo_results(frame, result):
    canvas = frame.copy()
    people = []

    if result.boxes is None or len(result.boxes) == 0:
        return canvas, people

    boxes = result.boxes.xyxy.detach().cpu().numpy()
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


def process_yolo_frame(rgb_frame):
    frame = np.asarray(rgb_frame, dtype=np.uint8)
    started = perf_counter()
    result = get_yolo_model().predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )[0]
    inference_ms = (perf_counter() - started) * 1000

    rendered, people = draw_yolo_results(frame, result)
    success, encoded = cv2.imencode(
        ".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 78]
    )
    if not success:
        raise RuntimeError("Could not encode the annotated frame.")

    import base64
    image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "people": people,
        "model_connected": True,
        "modelStatus": "YOLO Pose · TensorRT FP16",
        "message": "YOLO detection completed.",
        "frame_shape": list(frame.shape),
        "frame_dtype": str(frame.dtype),
        "annotated_frame": "data:image/jpeg;base64," + image_data,
        "inference_ms": round(inference_ms, 2),
        "fps": 1000.0 / inference_ms if inference_ms > 0 else 0.0,
    }
