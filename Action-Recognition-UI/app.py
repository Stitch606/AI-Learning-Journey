from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent


def process_frame(frame: Image.Image) -> dict:
    """
    MODEL INTEGRATION POINT.

    Replace this placeholder with your YOLO Pose -> keypoints -> ST-GCN++ call.
    Keep the return format compatible with ActionUI.setState():
      {
        "people": [
          {"id": 1, "person": "Person 1", "action": "Walking",
           "confidence": 0.93}
        ],
        "fps": 12.5,
        "logs": [{"text": "Person 1: Walking (0.93)"}]
      }
    The incoming frame is an RGB PIL image.
    """
    return {
        "people": [],
        "fps": 0.0,
        "logs": [],
        "model_connected": False,
        "message": "Frame received by Gradio. Connect your model in process_frame().",
    }


def receive_frame(frame: Image.Image) -> dict:
    if frame is None:
        return {"status": "error", "message": "No frame received."}

    result = process_frame(frame.convert("RGB"))
    return {
        "status": "ok",
        "width": frame.width,
        "height": frame.height,
        "received_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }


with gr.Blocks(title="Human Action Detection Bridge") as demo:
    frame_input = gr.Image(type="pil", visible=False)
    result_output = gr.JSON(visible=False)
    submit = gr.Button("Receive frame", visible=False)
    submit.click(
        fn=receive_frame,
        inputs=frame_input,
        outputs=result_output,
        api_name="receive_frame",
    )

app = FastAPI(title="Human Action Detection")
app.mount("/ui", StaticFiles(directory=ROOT, html=True), name="ui")
app = gr.mount_gradio_app(app, demo, path="/gradio")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
