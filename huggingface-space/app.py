import gradio as gr
from gpu import gpu_ready

custom_css = """
:root {
    --bg-color: #030712;
    --card-bg: rgba(15, 23, 42, 0.90);
    --border-glow: #00f0ff;
    --text-main: #f8fafc;
}

* {
    box-sizing: border-box !important;
}

html, body {
    width: 100% !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: var(--bg-color) !important;
}

body, .gradio-container {
    color: var(--text-main) !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 100% !important;
    height: 100dvh !important;
    min-height: 0 !important;
    overflow: hidden !important;
    padding: clamp(8px, 1.5vh, 16px) clamp(8px, 1.5vw, 20px) !important;
    margin: 0 !important;
    background: var(--bg-color) !important;
}

/* Keep the navigation at the top */
.nav-bar {
    width: 100% !important;
    max-width: 1500px !important;
    margin: 0 auto clamp(8px, 1.2vh, 14px) auto !important;
    gap: 12px !important;
    flex: 0 0 auto !important;
}

.nav-btn {
    min-height: clamp(42px, 6vh, 58px) !important;
    border: 1px solid var(--border-glow) !important;
    background: rgba(0, 240, 255, 0.12) !important;
    color: #fff !important;
    font-size: clamp(13px, 1.1vw, 17px) !important;
    font-weight: 700 !important;
    border-radius: 11px !important;
    box-shadow: 0 0 14px rgba(0, 240, 255, 0.20) !important;
}

.nav-btn:hover {
    background: rgba(0, 240, 255, 0.28) !important;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.38) !important;
}

/* One large shared landscape frame */
.cyber-frame {
    width: 100% !important;
    max-width: 1500px !important;
    width: min(100%, calc((100dvh - 112px) * 9 / 11)) !important;
    aspect-ratio: 9 / 11 !important;
    height: auto !important;
    min-height: 0 !important;
    flex: 0 0 auto !important;
    margin: 0 auto !important;
    padding: clamp(6px, 1vw, 14px) !important;
    overflow: hidden !important;
    background: var(--card-bg) !important;
    border: 2px solid var(--border-glow) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 28px rgba(0, 240, 255, 0.32),
                inset 0 0 22px rgba(0, 240, 255, 0.08) !important;
    backdrop-filter: blur(12px) !important;
}

/* Make Gradio's component fill the shared frame */
.display-area,
.display-area > div,
.display-area .wrap,
.display-area .container {
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    max-height: 100% !important;
}

.display-area {
    overflow: hidden !important;
}

.display-area video,
.display-area img {
    max-width: 100% !important;
    max-height: 100% !important;
    object-fit: contain !important;
}

/* Hide labels to preserve the usable display area */
.display-area label {
    display: none !important;
}

@media (max-width: 700px) {
    .gradio-container {
        padding: 8px !important;
    }

    .nav-bar {
        gap: 8px !important;
        margin-bottom: 8px !important;
    }

    .nav-btn {
        min-height: 44px !important;
        font-size: 13px !important;
    }

    .cyber-frame {
        width: min(100%, calc((100dvh - 78px) * 9 / 11)) !important;
        aspect-ratio: 9 / 11 !important;
        height: auto !important;
        padding: 5px !important;
        border-radius: 12px !important;
    }
}
"""

def show_live():
    return gr.update(visible=True), gr.update(visible=False)

def show_video():
    return gr.update(visible=False), gr.update(visible=True)

def preview_frame(frame):
    return frame

with gr.Blocks(css=custom_css) as demo:
    # Top navigation
    with gr.Row(elem_classes="nav-bar"):
        live_btn = gr.Button("📷 Live Camera", elem_classes="nav-btn")
        video_btn = gr.Button("🎬 Video", elem_classes="nav-btn")

    # Exactly one shared display frame
    with gr.Column(elem_classes="cyber-frame"):
        live_view = gr.Image(
            sources=["webcam"],
            streaming=True,
            type="numpy",
            label=None,
            height=720,
            visible=False,
            elem_classes="display-area"
        )

        video_view = gr.Video(
            sources=["upload"],
            label=None,
            interactive=True,
            height=720,
            visible=True,
            elem_classes="display-area"
        )

    live_btn.click(
        fn=show_live,
        outputs=[live_view, video_view],
        show_progress="hidden"
    )

    video_btn.click(
        fn=show_video,
        outputs=[live_view, video_view],
        show_progress="hidden"
    )

    # Receive the webcam stream continuously; this is the live-frame event.
    live_view.stream(
        fn=preview_frame,
        inputs=live_view,
        outputs=live_view,
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch()
