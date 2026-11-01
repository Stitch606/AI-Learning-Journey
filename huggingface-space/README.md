---
title: Human Action Detection
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
short_description: Real-time human action detection dashboard (interface only)
---

# Human Action Detection - interface

The page shows zeros until your code sends numbers. It does not process images and contains no AI.

## Files and what each one does

| File | Job | Imports from this project |
|---|---|---|
| `state.py` | Holds the numbers. Your code writes here | none |
| `view.py` | Turns the numbers into the page's HTML | none |
| `gpu.py` | ZeroGPU helper: the `@gpu` decorator | none |
| `style.css` | Colors and responsive layout | - |
| `app.py` | Connects the files, refreshes the page once a second, starts it | gpu, state, view |

Only `app.py` imports the other files. Your code imports `state` (and `gpu`, and `app` if it starts the page).

## Connect your code

Your code sends values with `push`. Only the names you pass are changed:

```python
from state import push

push(connected=True, model_loaded=True,
     people=3, actions=2, fps=24.1, runtime=elapsed_seconds, total=12,
     distribution={"Walking": 60, "Running": 20, "Standing": 10, "Sitting": 5, "Falling": 5},
     history=[(3, 1), (4, 2)],                       # one (people, actions) per second, last 60
     log=[("14:32:12", 3, "Person 3 detected - Running (0.92)")],
     rows=[(3, "Running", 0.92, "14:32:08")])
```

Allowed names: `connected`, `model`, `device`, `model_loaded`, `people`, `actions`, `fps`, `runtime`,
`total`, `distribution`, `history`, `log`, `rows`. Any other name raises an error, so typos show up immediately.

To start the page from your own file and keep your code running:

```python
import app
app.launch(block=False)
# your loop: push(...)
```

To run the page alone (it shows zeros): `python app.py`

## Run in Colab or on your computer

```
pip install gradio
python app.py
```

## Put it on a Hugging Face Space

1. Create a new Space, choose **Gradio** as the SDK. The page itself needs no GPU, so the free CPU hardware is enough.
2. Upload: `app.py`, `state.py`, `view.py`, `gpu.py`, `style.css`, `README.md`, `requirements.txt`.
   The block of settings at the top of this README (between the `---` lines) is what Hugging Face reads.
3. If you want a different Gradio version, change `sdk_version` in that block. Do **not** put `gradio` in `requirements.txt`; `sdk_version` installs it.
4. If your own code must run on the Space, make your own file (for example `run.py`) that imports `app`, starts your loop, and calls `app.launch()`. Then set `app_file: run.py` above.

### ZeroGPU

- Choose **ZeroGPU** as the Space hardware. `gpu.py` already contains one small `@gpu` function, because ZeroGPU refuses to start a Space with none ("No @spaces.GPU function detected during startup").
- Put `@gpu` on the functions of your code that need the GPU (see `gpu.py`). The GPU is given only while such a function runs, then released.
- Because of that, a background loop cannot hold the GPU. The model call should be a function with `@gpu` that is called from a Gradio event (a button, a camera frame, a file upload), not from a thread that runs forever.
- Outside ZeroGPU (Colab, your computer, normal Spaces) `@gpu` does nothing, so the same code runs everywhere.
