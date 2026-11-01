---
title: Human Action Detection
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
short_description: Responsive interactive human action detection dashboard
---

# Action Detector

A responsive Gradio dashboard with an interactive video input, webcam source selection, model controls, analytics tabs, event log, and downloadable configuration report.

## Current status

The interface is ready. The computer-vision processor is intentionally not connected yet, so the statistics remain at zero and the Start Analysis action validates the selected video and reports that the AI processor is waiting. No sample detections are presented as real model output.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Hugging Face Space entry point |
| `main.py` | Builds the Gradio app and wires UI events |
| `layout.py` | Dashboard, analytics, event log, and settings tabs |
| `styles.py` | Neon theme and phone/tablet responsive CSS |
| `callbacks.py` | Analysis adapter, reset, camera help, report export |
| `charts.py` | Plotly activity and distribution charts |
| `video_panel.py` | Video path handling and metadata inspection |
| `gpu.py` | Optional ZeroGPU decorator helper |

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Connect the vision model

Implement a processor function and pass it to `build_app(processor=your_processor)` from `main.py` or a custom entry point.

The processor is called with:

```python
processor(video_path=video_path, threshold=threshold, model=model, device=device)
```

It should return a dictionary with `people`, `actions`, `falls`, `collisions`, the four action percentages, aligned time-series arrays, and an `events` list containing action, confidence, and time.

## Notes

- Gradio is provided by the Hugging Face Gradio SDK version declared above; do not pin a second Gradio version in requirements.
- Video upload and webcam selection are provided by the Gradio video input.
- The downloadable CSV currently reports the selected video and model settings. It will include detection results once the processor is connected.
- The Reset Dashboard button clears displayed metrics; it does not interrupt an active model call.
- Free CPU hardware is sufficient to preview the interface. Choose suitable GPU hardware when the model is connected.
