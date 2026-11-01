import gradio as gr

def create_layout():
    with gr.Blocks(title="Action Detector") as demo:
        gr.HTML("""
        <header class="app-header"><div class="brand"><div class="brand-mark">🏃</div><div><div class="brand-title">ACTION <span>DETECTOR</span></div><div class="brand-subtitle">Detect · Track · Analyze · Stay Safe</div></div></div><div class="system-status"><span class="status-dot"></span><span>INTERFACE ONLINE</span><small>AI ENGINE WAITING</small></div></header>
        """)
        with gr.Tabs(elem_classes=["app-tabs"]):
            with gr.Tab("⌂  Dashboard", id="dashboard"):
                with gr.Row(equal_height=False, elem_classes=["top-grid"]):
                    with gr.Column(scale=7, min_width=0, elem_classes=["panel", "video-panel"]):
                        gr.HTML('<div class="section-heading"><span class="heading-icon">▣</span> LIVE VIDEO ANALYSIS <span class="live-pill">READY</span></div>')
                        video = gr.Video(label="", sources=["upload", "webcam"], interactive=True, height=360, elem_classes=["video-input"])
                        video_status = gr.Markdown("**Ready.** Upload a video or choose Webcam in the video source menu.", elem_classes=["status-message"])
                        progress = gr.Slider(0, 100, value=0, label="Analysis progress", interactive=False)
                        with gr.Row():
                            start_btn = gr.Button("▶  Start Analysis", variant="primary", elem_classes=["start-action"])
                            stop_btn = gr.Button("↺  Reset Dashboard", variant="stop", elem_classes=["stop-action"])
                    with gr.Column(scale=3, min_width=220, elem_classes=["panel", "stats-panel"]):
                        gr.HTML('<div class="section-heading">LIVE STATISTICS</div>')
                        with gr.Row():
                            people = gr.Number(0, label="👤 People", interactive=False)
                            actions = gr.Number(0, label="🏃 Actions", interactive=False)
                        with gr.Row():
                            falls = gr.Number(0, label="⚠️ Falls", interactive=False)
                            collisions = gr.Number(0, label="🛡️ Collisions", interactive=False)
                        gr.HTML('<div class="waiting-note">Statistics update when the vision model is connected.</div>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=0, elem_classes=["panel"]):
                        gr.HTML('<div class="section-heading">DETECTED ACTIONS</div>')
                        walking = gr.Slider(0, 100, value=0, label="🚶 Walking", interactive=False)
                        standing = gr.Slider(0, 100, value=0, label="🧍 Standing", interactive=False)
                        lifting = gr.Slider(0, 100, value=0, label="🏋️ Lifting", interactive=False)
                        falling = gr.Slider(0, 100, value=0, label="🚨 Fall", interactive=False)
                    with gr.Column(scale=1, min_width=0, elem_classes=["panel", "controls-panel"]):
                        gr.HTML('<div class="section-heading">VIDEO & MODEL CONTROLS</div>')
                        threshold = gr.Slider(0, 1, value=0.25, step=0.01, label="Confidence threshold")
                        with gr.Row():
                            model = gr.Dropdown(["YOLOv8", "YOLO11", "YOLO26"], value="YOLOv8", label="Detection model")
                            device = gr.Dropdown(["Auto (CPU)", "CPU", "CUDA"], value="Auto (CPU)", label="Device")
                        camera_btn = gr.Button("📷  Camera help", variant="secondary")
                        download_btn = gr.DownloadButton("⬇  Download analysis report", variant="secondary")
                        clear_btn = gr.Button("↺  Clear selected video", variant="secondary")
            with gr.Tab("▥  Analytics", id="analytics"):
                with gr.Row():
                    with gr.Column(elem_classes=["panel"], min_width=0):
                        gr.HTML('<div class="section-heading">ACTIONS OVER TIME</div>')
                        chart = gr.Plot(label="", show_label=False)
                    with gr.Column(elem_classes=["panel"], min_width=0):
                        gr.HTML('<div class="section-heading">ACTION DISTRIBUTION</div>')
                        distribution = gr.Plot(label="", show_label=False)
            with gr.Tab("☷  Event Log", id="events"):
                with gr.Column(elem_classes=["panel"]):
                    gr.HTML('<div class="section-heading">RECENT DETECTION EVENTS</div>')
                    events = gr.Dataframe(headers=["Action", "Confidence", "Time"], datatype=["str", "str", "str"], value=[], interactive=False, label="")
                    gr.Markdown("Events will appear here after the model processes a video.")
            with gr.Tab("⚙  Settings", id="settings"):
                with gr.Column(elem_classes=["panel"]):
                    gr.HTML('<div class="section-heading">APPLICATION SETTINGS</div>')
                    gr.Markdown("**Model connection**\n\nThe dashboard is in interface mode. Connect your vision processor through build_app(processor=...).")
                    gr.Markdown("**Available controls**\n\nConfidence threshold, model selection, and device selection are passed to the processor callback.")
                    gr.Markdown("**Responsive layout**\n\nPanels rearrange into a single-column layout on phone screens.")
    return {
        "demo": demo, "video": video, "video_status": video_status, "progress": progress,
        "people": people, "actions": actions, "falls": falls, "collisions": collisions,
        "walking": walking, "standing": standing, "lifting": lifting, "falling": falling,
        "start_btn": start_btn, "stop_btn": stop_btn, "download_btn": download_btn,
        "camera_btn": camera_btn, "clear_btn": clear_btn, "threshold": threshold,
        "model": model, "device": device, "chart": chart, "distribution": distribution,
        "events": events
    }
