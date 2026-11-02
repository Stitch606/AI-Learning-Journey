import gradio as gr
from gpu import gpu_ready

custom_css = """
:root {
    --bg-color: #030712;
    --card-bg: rgba(15, 23, 42, 0.88);
    --border-glow: #00f0ff;
    --text-main: #f8fafc;
}

html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: var(--bg-color) !important;
    color: var(--text-main) !important;
    overflow-x: hidden !important;
}

.gradio-container {
    min-height: 100svh !important;
    max-width: 100% !important;
    padding: 14px 12px !important;
    margin: 0 auto !important;
    background: var(--bg-color) !important;
    color: var(--text-main) !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

/* Hide page elements that consume vertical space */
footer, header {
    display: none !important;
}

/* Top navigation */
.nav-bar {
    width: 100% !important;
    max-width: 720px !important;
    margin: 0 auto 12px auto !important;
    gap: 10px !important;
}

.nav-btn {
    min-height: 44px !important;
    border: 1px solid var(--border-glow) !important;
    background: rgba(0, 240, 255, 0.12) !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 0 12px rgba(0, 240, 255, 0.18) !important;
    transition: background 0.2s ease, box-shadow 0.2s ease !important;
}

.nav-btn:hover {
    background: rgba(0, 240, 255, 0.28) !important;
    box-shadow: 0 0 18px rgba(0, 240, 255, 0.35) !important;
}

/* One shared landscape display frame */
.cyber-frame {
    width: 100% !important;
    max-width: 720px !important;
    aspect-ratio: 16 / 9 !important;
    box-sizing: border-box !important;
    margin: 0 auto !important;
    padding: 10px !important;
    overflow: hidden !important;
    background: var(--card-bg) !important;
    border: 2px solid var(--border-glow) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.30),
                inset 0 0 20px rgba(0, 240, 255, 0.08) !important;
    backdrop-filter: blur(12px) !important;
}

/* Both modes occupy the same frame dimensions */
.display-area {
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

.display-area > div,
.display-area .wrap,
.display-area .container {
    height: 100% !important;
    min-height: 0 !important;
}

/* Remove excess labels and component spacing */
.display-area label {
    display: none !important;
}

@media (max-width: 600px) {
    .gradio-container {
        padding: 10px 8px !important;
    }
    .nav-bar {
        margin-bottom: 8px !important;
        gap: 8px !important;
    }
    .nav-btn {
        min-height: 40px !important;
        font-size: 13px !important;
    }
    .cyber-frame {
        padding: 6px !important;
        border-radius: 12px !important;
    }
}
"""

with gr.Blocks(css=custom_css) as demo:
    # Navigation bar
    with gr.Row(elem_classes="nav-bar"):
        live_btn = gr.Button(
            "📷 Live Camera",
            elem_classes="nav-btn"
        )
        video_btn = gr.Button(
            "🎬 Video",
            elem_classes="nav-btn"
        )

    # A single landscape frame; only one mode is visible at a time
    with gr.Column(elem_classes="cyber-frame"):
        live_view = gr.Image(
            sources=["webcam"],
            streaming=True,
            type="numpy",
            label=None,
            height=405,
            visible=False,
            elem_classes="display-area"
        )

        video_view = gr.Video(
            sources=["upload"],
            label=None,
            interactive=True,
            height=405,
            visible=True,
            elem_classes="display-area"
        )

    # Switch the content inside the same frame
    live_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[live_view, video_view]
    )

    video_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[live_view, video_view]
    )

if __name__ == "__main__":
    demo.launch()
