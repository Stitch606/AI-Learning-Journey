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
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Column(elem_classes="cyber-frame"):
        video_output = gr.Video(label="Cyber Video Stream", interactive=False)

        with gr.Row():
            upload_btn = gr.Button("📁 Upload Video", elem_classes="cyber-btn")
            camera_btn = gr.Button("📷 Open Camera", elem_classes="cyber-btn")

    with gr.Row():
        camera_choice = gr.Radio(
            choices=["Front Camera", "Rear Camera"],
            value="Front Camera",
            label="Switch Camera",
        )

if __name__ == "__main__":
    demo.launch()
