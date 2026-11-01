import gradio as gr


def create_layout():
    with gr.Blocks(title="Action Detector", elem_classes=["action-app"]) as demo:
        gr.HTML("""
        <header class="hero-header">
          <div class="hero-brand">
            <div class="brand-symbol"><span>AD</span><i></i></div>
            <div class="brand-copy">
              <div class="eyebrow">COMPUTER VISION PLATFORM</div>
              <div class="brand-title">ACTION<span> DETECTOR</span></div>
              <div class="brand-subtitle">Understand movement. Detect risk. See every moment.</div>
            </div>
          </div>
          <div class="header-meta">
            <div class="engine-chip"><span class="pulse-dot"></span><span>ENGINE STANDBY</span></div>
            <div class="version-label">VISION WORKSPACE <b>01</b></div>
          </div>
        </header>
        <div class="welcome-row">
          <div><div class="eyebrow">WORKSPACE / OVERVIEW</div><h1>Activity dashboard</h1><p>Upload a clip or connect a camera to begin your analysis.</p></div>
          <div class="workspace-tag"><span class="tiny-square"></span> LOCAL SESSION</div>
        </div>
        """)
        with gr.Tabs(elem_classes=["app-tabs"]):
            with gr.Tab("◫  Overview", id="dashboard"):
                with gr.Row(equal_height=False, elem_classes=["top-grid"]):
                    with gr.Column(scale=7, min_width=0, elem_classes=["panel", "video-panel"]):
                        gr.HTML('<div class="panel-topline"><div><div class="eyebrow">INPUT SOURCE</div><div class="section-title">Video monitor</div></div><span class="state-pill">● READY</span></div>')
                        video = gr.Video(label="", sources=["upload", "webcam"], interactive=True, height=390, elem_classes=["video-input"])
                        video_status = gr.Markdown("**Ready for input.** Upload a video or choose Webcam from the source selector.", elem_classes=["status-message"])
                        with gr.Row(elem_classes=["video-actions"]):
                            start_btn = gr.Button("▶  Run analysis", variant="primary", elem_classes=["start-action"])
                            stop_btn = gr.Button("↺  Reset", variant="secondary", elem_classes=["stop-action"])
                        progress = gr.Slider(0, 100, value=0, label="PROCESSING PROGRESS", interactive=False, elem_classes=["progress-control"])
                    with gr.Column(scale=5, min_width=250, elem_classes=["panel", "insights-panel"]):
                        gr.HTML('<div class="panel-topline"><div><div class="eyebrow">LIVE METRICS</div><div class="section-title">Detection summary</div></div><span class="metric-caption">SESSION</span></div>')
                        with gr.Row(elem_classes=["metric-row"]):
                            people = gr.Number(0, label="PEOPLE TRACKED", interactive=False, elem_classes=["metric-card", "metric-cyan"])
                            actions = gr.Number(0, label="ACTIONS FOUND", interactive=False, elem_classes=["metric-card", "metric-violet"])
                        with gr.Row(elem_classes=["metric-row"]):
                            falls = gr.Number(0, label="FALL ALERTS", interactive=False, elem_classes=["metric-card", "metric-rose"])
                            collisions = gr.Number(0, label="COLLISIONS", interactive=False, elem_classes=["metric-card", "metric-amber"])
                        gr.HTML("""
                        <div class="insight-note">
                          <div class="note-icon">✦</div><div><b>Waiting for your model</b><span>Metrics will populate when the vision processor is connected.</span></div>
                        </div>
                        <div class="mini-divider"></div>
                        <div class="quick-facts"><div><span>INPUT</span><b>Video / Webcam</b></div><div><span>OUTPUT</span><b>Actions & events</b></div></div>
                        """)
                with gr.Row(equal_height=False, elem_classes=["lower-grid"]):
                    with gr.Column(scale=6, min_width=0, elem_classes=["panel", "action-panel"]):
                        gr.HTML('<div class="panel-topline"><div><div class="eyebrow">CLASSIFICATION</div><div class="section-title">Action breakdown</div></div><span class="metric-caption">CONFIDENCE</span></div>')
                        walking = gr.Slider(0, 100, value=0, label="🚶  Walking", interactive=False)
                        standing = gr.Slider(0, 100, value=0, label="🧍  Standing", interactive=False)
                        lifting = gr.Slider(0, 100, value=0, label="🏋  Lifting", interactive=False)
                        falling = gr.Slider(0, 100, value=0, label="⚠  Fall", interactive=False)
                    with gr.Column(scale=6, min_width=0, elem_classes=["panel", "controls-panel"]):
                        gr.HTML('<div class="panel-topline"><div><div class="eyebrow">CONFIGURATION</div><div class="section-title">Analysis controls</div></div><span class="metric-caption">ADJUSTABLE</span></div>')
                        threshold = gr.Slider(0, 1, value=0.25, step=0.01, label="CONFIDENCE THRESHOLD")
                        with gr.Row(elem_classes=["select-row"]):
                            model = gr.Dropdown(["YOLOv8", "YOLO11", "YOLO26"], value="YOLOv8", label="DETECTION MODEL")
                            device = gr.Dropdown(["Auto (CPU)", "CPU", "CUDA"], value="Auto (CPU)", label="PROCESSING DEVICE")
                        with gr.Row(elem_classes=["utility-actions"]):
                            camera_btn = gr.Button("▣  Camera help", variant="secondary")
                            clear_btn = gr.Button("⌫  Clear video", variant="secondary")
                        download_btn = gr.DownloadButton("↓  Export session report", variant="secondary", elem_classes=["download-action"])
            with gr.Tab("▤  Analytics", id="analytics"):
                with gr.Row(equal_height=False, elem_classes=["analytics-grid"]):
                    with gr.Column(elem_classes=["panel"], min_width=0):
                        gr.HTML('<div class="panel-topline"><div><div class="eyebrow">TEMPORAL ANALYSIS</div><div class="section-title">Actions over time</div></div></div>')
                        chart = gr.Plot(label="", show_label=False)
                    with gr.Column(elem_classes=["panel"], min_width=0):
                        gr.HTML('<div class="panel-topline"><div><div class="eyebrow">CLASS DISTRIBUTION</div><div class="section-title">Action distribution</div></div></div>')
                        distribution = gr.Plot(label="", show_label=False)
            with gr.Tab("☷  Event log", id="events"):
                with gr.Column(elem_classes=["panel", "events-panel"]):
                    gr.HTML('<div class="panel-topline"><div><div class="eyebrow">DETECTION HISTORY</div><div class="section-title">Recent events</div></div><span class="metric-caption">LATEST FIRST</span></div>')
                    events = gr.Dataframe(headers=["Action", "Confidence", "Time"], datatype=["str", "str", "str"], value=[], interactive=False, label="")
                    gr.Markdown("Events will appear here after the model processes a video.", elem_classes=["empty-events"])
            with gr.Tab("⚙  Settings", id="settings"):
                with gr.Column(elem_classes=["panel", "settings-panel"]):
                    gr.HTML('<div class="eyebrow">WORKSPACE PREFERENCES</div><div class="section-title">Application settings</div>')
                    gr.Markdown("**Model connection**\n\nThe dashboard is ready for your vision processor. Connect it through build_app(processor=...).")
                    gr.Markdown("**Analysis parameters**\n\nConfidence threshold, model selection, and processing device are passed to the processor callback.")
                    gr.Markdown("**Responsive interface**\n\nThe workspace rearranges its panels for phone and tablet screens.")
        gr.HTML('<footer class="app-footer"><span> ACTION DETECTOR <b> / </b> COMPUTER VISION WORKSPACE</span><span>BUILT FOR REAL-TIME INSIGHT <i>✦</i></span></footer>')
    return {
        "demo": demo, "video": video, "video_status": video_status, "progress": progress,
        "people": people, "actions": actions, "falls": falls, "collisions": collisions,
        "walking": walking, "standing": standing, "lifting": lifting, "falling": falling,
        "start_btn": start_btn, "stop_btn": stop_btn, "download_btn": download_btn,
        "camera_btn": camera_btn, "clear_btn": clear_btn, "threshold": threshold,
        "model": model, "device": device, "chart": chart, "distribution": distribution,
        "events": events
    }
