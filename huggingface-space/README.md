---
title: Human Action Detection
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
short_description: Custom HTML/CSS action detection frontend powered by Gradio Server
---

# Human Action Detection

A custom web interface served by Gradio Server. The frontend is built with HTML, CSS, and JavaScript; Python endpoints are provided by Gradio's API engine and can use Hugging Face ZeroGPU.

## Project files

- `app.py`: Gradio Server, custom routes, and queued Python API.
- `templates/index.html`: Web interface.
- `static/style.css`: Responsive interface styling.
- `static/app.js`: Camera/video controls and Gradio JavaScript client.
- `gpu.py`: ZeroGPU decorator helper and startup GPU registration.
- `requirements.txt`: Additional Python dependencies.

## Current status

The interface supports camera preview and video upload. The analysis endpoint is a connection placeholder; the YOLO/action-recognition pipeline has not yet been connected.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Gradio 6.31.0 is provided by the Hugging Face Space SDK version declared above.
