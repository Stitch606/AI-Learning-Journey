---
title: Human Action Detection
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.0.0
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
processor(
    video_path=video_path,
    threshold=threshold,
    model=model,
    device=device
)
```

It should return a dictionary with these fields:

```python
{
    "people": 0,
    "actions": 0,
    "falls": 0,
    "collisions": 0,
    "walking": 0,
    "standing": 0,
    "lifting": 0,
    "falling": 0,
    "timestamps": [],
    "walking_history": [],
    "standing_history": [],
    "lifting_history": [],
    "falling_history": [],
    "events": [
        {"action": "Walking", "confidence": 0.92, "time": "14:32:12"}
    ]
}
```

Action values are percentages from 0 to 100. History arrays should align with `timestamps`. Event entries use `action`, `confidence`, and `time`.

The current callback processes a selected video when the processor is supplied. For a live streaming pipeline, we will add a frame-by-frame streaming callback when the model interface is available.

## Notes

- Video upload and webcam selection are provided by the Gradio video input.
- The downloadable CSV currently reports the selected video and model settings. It will include detection results once the processor is connected.
- The Reset Dashboard button clears displayed metrics; it does not interrupt an active model call.
- Free CPU hardware is sufficient to preview the interface. Choose suitable GPU hardware when the model is connected.
