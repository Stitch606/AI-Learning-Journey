---
title: Human Action Detection
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
short_description: Custom action detection web interface
---

# Human Action Detection

A custom HTML/CSS/JavaScript dashboard served by **Gradio Server**. Gradio remains the application server, API bridge, queue, and ZeroGPU integration.

## Layout

- `app.py`: Gradio Server, frontend routes, and frame API.
- `gpu.py`: ZeroGPU decorator helper and startup GPU registration.
- `templates/index.html`: Main HTML page.
- `static/styles/style.css`: All interface styling.
- `static/scripts/state.js`: UI state and theme state.
- `static/scripts/app.js`: UI behavior, camera/video capture, and Gradio client.
- `process/`: Model implementation and weights. Kept separate from the frontend.
- `requirements.txt`: Additional Python dependencies.

## Frame flow

1. The browser captures a camera or video frame and resizes it before transfer.
2. The JavaScript Gradio client sends the frame to `/receive_frame`.
3. Gradio receives a PIL image; `process_frame()` converts it to an RGB NumPy array.
4. The endpoint returns a result payload to the browser.

The frame endpoint currently validates the Gradio bridge and image conversion. The YOLO Pose → ST-GCN++ inference pipeline is the next integration step.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.
