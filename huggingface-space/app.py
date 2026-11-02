import gradio as gr
from gpu import gpu_ready

custom_css = """
:root {
    --bg-color: #030712;
    --card-bg: rgba(15, 23, 42, 0.85);
    --border-glow: #00f0ff;
    --accent-purple: #a855f7;
    --text-main: #f8fafc;
}

body, .gradio-container {
    background-color: var(--bg-color) !important;
    color: var(--text-main) !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

.cyber-frame {
    background: var(--card-bg) !important;
    border: 2px solid var(--border-glow) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.35), inset 0 0 20px rgba(0, 240, 255, 0.1) !important;
    backdrop-filter: blur(12px) !important;
    padding: 20px !important;
}

.cyber-btn {
    border: 1px solid var(--border-glow) !important;
    background: rgba(0, 240, 255, 0.15) !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 0 12px rgba(0, 240, 255, 0.25) !important;
    transition: all 0.2s ease !important;
}

.cyber-btn:hover {
    background: var(--border-glow) !important;
    color: #000 !important;
    box-shadow: 0 0 20px var(--border-glow) !important;
}

.video-panel {
    border: 1px solid rgba(0, 240, 255, 0.35) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

.camera-note {
    color: #a5b4fc !important;
    font-size: 0.9rem !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    with gr.Column(elem_classes="cyber-frame"):
        gr.Markdown("## Cyber Video Stream")
        gr.Markdown(
            "Upload a video or record a clip with your device camera.",
            elem_classes="camera-note",
        )

        with gr.Tabs():
            with gr.Tab("Upload Video"):
                uploaded_video = gr.Video(
                    label="Select a video",
                    sources=["upload"],
                    format="mp4",
                    interactive=True,
                    height=420,
                    elem_classes="video-panel",
                )

            with gr.Tab("Camera"):
                camera_choice = gr.Radio(
                    choices=["Front Camera", "Rear Camera"],
                    value="Front Camera",
                    label="Choose camera",
                )

                front_camera = gr.Video(
                    label="Front Camera",
                    sources=["webcam"],
                    webcam_options=gr.WebcamOptions(
                        mirror=True,
                        constraints={"facingMode": "user"},
                    ),
                    format="mp4",
                    interactive=True,
                    height=420,
                    elem_classes="video-panel",
                    visible=True,
                )

                rear_camera = gr.Video(
                    label="Rear Camera",
                    sources=["webcam"],
                    webcam_options=gr.WebcamOptions(
                        mirror=False,
                        constraints={"facingMode": {"exact": "environment"}},
                    ),
                    format="mp4",
                    interactive=True,
                    height=420,
                    elem_classes="video-panel",
                    visible=False,
                )

                camera_choice.change(
                    fn=lambda choice: (
                        gr.update(visible=choice == "Front Camera"),
                        gr.update(visible=choice == "Rear Camera"),
                    ),
                    inputs=camera_choice,
                    outputs=[front_camera, rear_camera],
                )

        gr.Markdown(
            "Select a camera, allow browser permission, then use the camera recorder to capture a video."
        )

if __name__ == "__main__":
    demo.launch()
