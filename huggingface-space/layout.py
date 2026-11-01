
import gradio as gr


def create_layout():

    with gr.Blocks(
        title="Action Detector"
    ) as demo:

        # ==================================
        # HEADER
        # ==================================

        gr.HTML("""
        <header class="app-header">

            <div class="brand">

                <div class="brand-icon">🏃</div>

                <div>
                    <div class="brand-title">
                        ACTION <span>DETECTOR</span>
                    </div>

                    <div class="brand-subtitle">
                        Detect · Track · Analyze · Stay Safe
                    </div>
                </div>

            </div>

            <div class="system-status">
                🟢 System Online
                <div style="font-size:11px;color:#91a9ce">
                    YOLOv8 + Pose
                </div>
            </div>

        </header>
        """)

        # ==================================
        # MAIN LAYOUT
        # ==================================

        with gr.Row(elem_classes=["main-layout"]):

            # ==================================
            # SIDEBAR
            # ==================================

            with gr.Column(
                scale=1,
                min_width=120,
                elem_classes=["sidebar"]
            ):

                gr.HTML("""
                <div class="sidebar-item active">
                    🏠 Dashboard
                </div>
                """)

                gr.HTML("""
                <div class="sidebar-item">
                    🎥 Video Analysis
                </div>
                """)

                gr.HTML("""
                <div class="sidebar-item">
                    📊 Analytics
                </div>
                """)

                gr.HTML("""
                <div class="sidebar-item">
                    📋 Event Log
                </div>
                """)

                gr.HTML("""
                <div class="sidebar-item">
                    ⚙️ Settings
                </div>
                """)

            # ==================================
            # DASHBOARD CONTENT
            # ==================================

            with gr.Column(
                scale=6,
                min_width=0,
                elem_classes=["dashboard-content"]
            ):

                # ==================================
                # TOP SECTION
                # VIDEO + STATISTICS
                # ==================================

                with gr.Row(equal_height=False):

                    # VIDEO PANEL

                    with gr.Column(
                        scale=7,
                        min_width=0,
                        elem_classes=["panel", "video-panel"]
                    ):

                        gr.HTML("""
                        <div class="panel-title">
                            🎥 LIVE VIDEO ANALYSIS
                        </div>
                        """)

                        video = gr.Video(
                            label="",
                            sources=["upload", "webcam"],
                            interactive=False,
                            elem_classes=["video-frame"]
                        )

                        with gr.Row():
                            video_status = gr.Markdown(
                                "🟢 Ready for analysis",
                                elem_classes=["video-status"]
                            )

                        # VIDEO TIMELINE

                        progress = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="Video Progress",
                            interactive=False
                        )

                    # LIVE STATISTICS

                    with gr.Column(
                        scale=3,
                        min_width=0,
                        elem_classes=["panel"]
                    ):

                        gr.HTML("""
                        <div class="panel-title">
                            LIVE STATISTICS
                        </div>
                        """)

                        with gr.Row():
                            people = gr.Number(
                                value=0,
                                label="👤 Total People",
                                interactive=False
                            )

                            actions = gr.Number(
                                value=0,
                                label="🏃 Actions Detected",
                                interactive=False
                            )

                        with gr.Row():
                            falls = gr.Number(
                                value=0,
                                label="⚠️ Fall Detected",
                                interactive=False
                            )

                            collisions = gr.Number(
                                value=0,
                                label="🛡️ Collisions",
                                interactive=False
                            )

                # ==================================
                # DETECTED ACTIONS
                # ==================================

                with gr.Row():

                    with gr.Column(
                        scale=1,
                        elem_classes=["panel"]
                    ):

                        gr.HTML("""
                        <div class="panel-title">
                            DETECTED ACTIONS
                        </div>
                        """)

                        walking = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="🚶 Walking (%)",
                            interactive=False
                        )

                        standing = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="🧍 Standing (%)",
                            interactive=False
                        )

                        lifting = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="🏋️ Lifting (%)",
                            interactive=False
                        )

                        falling = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="🚨 Fall (%)",
                            interactive=False
                        )

                # ==================================
                # ANALYSIS ACTIONS
                # ==================================

                with gr.Row():

                    start_btn = gr.Button(
                        "▶ Start Analysis",
                        variant="primary"
                    )

                    stop_btn = gr.Button(
                        "■ Stop Analysis",
                        variant="stop"
                    )

                    download_btn = gr.DownloadButton(
                        "⬇ Download Result"
                    )

                # ==================================
                # BOTTOM SECTION
                # ==================================

                with gr.Row(equal_height=False):

                    # VIDEO CONTROLS

                    with gr.Column(
                        scale=1,
                        elem_classes=["panel"]
                    ):

                        gr.HTML("""
                        <div class="panel-title">
                            VIDEO CONTROLS
                        </div>
                        """)

                        upload = gr.File(
                            label="Upload Video",
                            file_types=["video"]
                        )

                        camera_btn = gr.Button(
                            "📷 Use Camera"
                        )

                        threshold = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.25,
                            step=0.01,
                            label="Confidence Threshold"
                        )

                        model = gr.Dropdown(
                            choices=[
                                "YOLOv8",
                                "YOLO11",
                                "YOLO26"
                            ],
                            value="YOLOv8",
                            label="Model"
                        )

                        device = gr.Dropdown(
                            choices=[
                                "Auto (CPU)",
                                "CPU",
                                "CUDA"
                            ],
                            value="Auto (CPU)",
                            label="Device"
                        )

                    # ACTIONS OVER TIME

                    with gr.Column(
                        scale=1,
                        elem_classes=["panel", "chart-panel"]
                    ):

                        gr.HTML("""
                        <div class="panel-title">
                            ACTIONS OVER TIME
                        </div>
                        """)

                        chart = gr.Plot(
                            label="",
                            show_label=False
                        )

                    # RECENT EVENTS

                    with gr.Column(
                        scale=1,
                        elem_classes=["panel"]
                    ):

                        gr.HTML("""
                        <div class="panel-title">
                            RECENT EVENTS
                        </div>
                        """)

                        events = gr.Dataframe(
                            headers=[
                                "Action",
                                "Confidence",
                                "Time"
                            ],
                            datatype=[
                                "str",
                                "str",
                                "str"
                            ],
                            value=[],
                            interactive=False,
                            label=""
                        )

    return {
        "demo": demo,

        # Video
        "video": video,
        "video_status": video_status,
        "progress": progress,

        # Statistics
        "people": people,
        "actions": actions,
        "falls": falls,
        "collisions": collisions,

        # Action percentages
        "walking": walking,
        "standing": standing,
        "lifting": lifting,
        "falling": falling,

        # Buttons
        "start_btn": start_btn,
        "stop_btn": stop_btn,
        "download_btn": download_btn,
        "camera_btn": camera_btn,

        # Settings
        "upload": upload,
        "threshold": threshold,
        "model": model,
        "device": device,

        # Analytics
        "chart": chart,
        "events": events
    }
