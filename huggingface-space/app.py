from pathlib import Path

import gradio as gr
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from gpu import gpu, gpu_ready


BASE_DIR = Path(__file__).parent

# Gradio Server keeps the Gradio API engine, queue, and ZeroGPU integration
# while allowing a completely custom HTML/CSS/JS frontend.
app = gr.Server()

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
    return {"status": "ok", "frontend": "ready", "model": "not_connected"}


@app.api(name="analyze_video")
@gpu
def analyze_video(video_path: str):
    """
    Frontend-to-Gradio API connection.

    Replace this response with the real detection pipeline when the model
    and its preprocessing/postprocessing code are connected.
    """
    return {
        "status": "model_not_connected",
        "message": "The web interface is running, but the detection pipeline has not been connected yet.",
        "received_video": bool(video_path),
    }


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
