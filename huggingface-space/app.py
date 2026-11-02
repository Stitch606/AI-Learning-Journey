import gradio as gr

# Custom CSS for the cyber neon design and fixed indigo background
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

/* Neon video frame with preserved dimensions and stability */
.cyber-frame {
    background: var(--card-bg) !important;
    border: 2px solid var(--border-glow) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.35), inset 0 0 20px rgba(0, 240, 255, 0.1) !important;
    backdrop-filter: blur(12px) !important;
    padding: 16px !important;
    max-width: 600px !important;
    margin: 0 auto !important;
}

/* Cyber neon buttons */
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
        # Video player or live display inside the same frame
        video_output = gr.Video(
            label="Cyber Video Stream",
            interactive=False
        )

        # Controls: upload a video or open the front camera
        with gr.Row():
            upload_input = gr.File(
                label="Upload Video",
                file_types=["video"],
                elem_classes="cyber-btn"
            )
            camera_btn = gr.Button(
                "📷 Live Camera",
                elem_classes="cyber-btn"
            )

        # Display the uploaded video immediately in the video player
        upload_input.change(fn=lambda x: x, inputs=upload_input, outputs=video_output)

if __name__ == "__main__":
    demo.launch()
