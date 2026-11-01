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
          <div><div class="eyebrow">WORKSPACE / OVERVIEW</div><h1>Activity dashboard</h1><p>Your video analysis workspace, with every control in reach.</p></div>
          <div class="workspace-tag"><span class="tiny-square"></span> LOCAL SESSION</div>
        </div>
        """)
        with gr.Tabs(elem_classes=["app-tabs"]):
            with gr.Tab("◫  Overview", id="dashboard"):
                # Primary workspace: video owns the visual focus.
                with gr.Row(equal_height=False, elem_classes=["stage-grid"]):
                    with gr.Column(scale=8, min_width=0, elem_classes=["panel", "video-panel"]):
                        gr.HTML('<div class="panel-topline"><div><div class="eyebrow">01 / INPUT SOURCE</div><div class="section-title">Video monitor</div></div><span class="state-pill">● READY</span></div>')
                        video = gr.Video(label="", sources=["upload", "webcam"], interactive=True, height=460, elem_classes=["video-input"])
                        video_status = gr.Markdown("**Ready for input.** Upload a video or choose Webcam from the source selector.", elem_classes=["status-message"])
                        with gr.Row(elem_classes=["video-actions"]):
                            start_btn = gr.Button("▶  Run analysis", variant="primary", elem_classes=["start-action"])
                            stop_btn = gr.Button("↺  Reset session", variant="secondary", elem_classes=["stop-action"])
                        progress = gr.Slider(0, 100, value=0, label="PROCESSING PROGRESS", interactive=False, elem_classes=["progress-control"])
                    with gr.Column(scale=4, min_width=250, elem_classes=["panel", "control-rail"]):
                        gr.HTML('<div class="panel-topline"><div><div class="eyebrow">02 / CONFIGURATION</div><div class="section-title">Analysis controls</div></div><span class="metric-caption">SETUP</span></div>')
                        threshold = gr.Slider(0, 1, value=0.25, step=0.01, label="CONFIDENCE THRESHOLD")
                        model = gr.Dropdown(["YOLOv8", "YOLO11", "YOLO26"], value="YOLOv8", label="DETECTION MODEL")
                        device = gr.Dropdown(["Auto (CPU)", "CPU", "CUDA"], value="Auto (CPU)", label="PROCESSING DEVICE")
                        gr.HTML('<div class="rail-divider"></div><div class="eyebrow">QUICK ACTIONS</div>')
                        with gr.Column(elem_classes=["rail-actions"]):
                            camera_btn = gr.Button("▣  Camera help", variant="secondary")
                            clear_btn = gr.Button("⌫  Clear video", variant="secondary")
                            download_btn = gr.DownloadButton("↓  Export session report", variant="secondary", elem_classes=["download-action"])
                        gr.HTML("""
                        <div class="rail-tip"><span class="tip-mark">✦</span><div><b>Ready when you are</b><p>Select a source, tune the confidence threshold, then run the analysis.</p></div></div>
                        """)
                # Full-width metrics strip, separate from the video stage.
                with gr.Column(elem_classes=["metrics-section"]):
                    gr.HTML('<div class="section-heading"><div><div class="eyebrow">03 / SESSION SNAPSHOT</div><div class="section-title">Detection summary</div></div><span class="metric-caption">LIVE METRICS</span></div>')
                    with gr.Row(elem_classes=["metric-strip"]):
                        people = gr.Number(0, label="PEOPLE TRACKED", interactive=False, elem_classes=["metric-card", "metric-cyan"])
                        actions = gr.Number(0, label="ACTIONS FOUND", interactive=False, elem_classes=["metric-card", "metric-violet"])
                        falls = gr.Number(0, label="FALL ALERTS", interactive=False, elem_classes=["metric-card", "metric-rose"])
                        collisions = gr.Number(0, label="COLLISIONS", interactive=False, elem_classes=["metric-card", "metric-amber"])
                with gr.Row(equal_height=False, elem_classes=["lower-grid"]):
                    with gr.Column(scale=6, min_width=0, elem_classes=["panel", "action-panel"]):
                        gr.HTML('<div class="panel-topline"><div><div class="eyebrow">04 / CLASSIFICATION</div><div class="section-title">Action breakdown</div></div><span class="metric-caption">CONFIDENCE</span></div>')
                        walking = gr.Slider(0, 100, value=0, label="🚶  Walking", interactive=False)
                        standing = gr.Slider(0, 100, value=0, label="🧍  Standing", interactive=False)
                        lifting = gr.Slider(0, 100, value=0, label="🏋  Lifting", interactive=False)
                        falling = gr.Slider(0, 100, value=0, label="⚠  Fall", interactive=False)
                    with gr.Column(scale=6, min_width=0, elem_classes=["panel", "session-panel"]):
                        gr.HTML('<div class="panel-topline"><div><div class="eyebrow">05 / SESSION INFO</div><div class="section-title">Analysis at a glance</div></div></div>')
                        gr.HTML("""
                        <div class="session-info-grid">
                          <div class="session-info"><span>INPUT</span><b>Video / Webcam</b><small>Local media source</small></div>
                          <div class="session-info"><span>OUTPUT</span><b>Actions & events</b><small>Detection results</small></div>
                          <div class="session-info"><span>MODEL</span><b>YOLO family</b><small>Selectable detector</small></div>
                          <div class="session-info"><span>STATUS</span><b>Awaiting processor</b><small>Connect your AI pipeline</small></div>
                        </div>
                        """)
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
        gr.HTML('<div class="app-footer"><span>ACTION DETECTOR <b> / </b> COMPUTER VISION WORKSPACE</span><span>BUILT FOR REAL-TIME INSIGHT <i>✦</i></span></div>')
    return {
        "demo": demo, "video": video, "video_status": video_status, "progress": progress,
        "people": people, "actions": actions, "falls": falls, "collisions": collisions,
        "walking": walking, "standing": standing, "lifting": lifting, "falling": falling,
        "start_btn": start_btn, "stop_btn": stop_btn, "download_btn": download_btn,
        "camera_btn": camera_btn, "clear_btn": clear_btn, "threshold": threshold,
        "model": model, "device": device, "chart": chart, "distribution": distribution,
        "events": events
    }
