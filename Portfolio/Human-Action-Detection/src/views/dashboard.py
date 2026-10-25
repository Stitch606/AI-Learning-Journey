import flet as ft


def build_dashboard(page: ft.Page):
    """Starter dashboard entry point for the Human Action Detection UI.

    Connect your inference pipeline here: send video frames to the detector,
    then update the preview, detected action, confidence, and event log.
    """
    page.padding = 24
    page.scroll = ft.ScrollMode.AUTO

    status = ft.Text("Waiting for video input", color=ft.Colors.BLUE_200)
    action = ft.Text("No action detected", size=24, weight=ft.FontWeight.BOLD)
    confidence = ft.Text("Confidence: —")
    preview = ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.Icons.VIDEO_CAMERA_BACK_OUTLINED, size=64),
                ft.Text("Video preview", size=20),
                ft.Text("Connect your YOLO / action-classification pipeline to display frames here."),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        alignment=ft.Alignment(0, 0),
        expand=True,
        height=300,
        border=ft.Border.all(1, ft.Colors.BLUE_GREY_400),
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.BLUE_GREY_500),
    )

    def choose_video(e):
        status.value = "Video selection needs a file-picker handler for your target platform."
        page.update()

    page.add(
        ft.Row(
            [
                ft.Column(
                    [
                        ft.Text("HUMAN ACTION DETECTION", size=12, color=ft.Colors.BLUE_300),
                        ft.Text("Monitoring Dashboard", size=30, weight=ft.FontWeight.BOLD),
                        ft.Text("Video-based human action analysis"),
                    ],
                    expand=True,
                ),
                ft.Container(content=ft.Text("DEMO UI"), padding=10, border_radius=8,
                             bgcolor=ft.Colors.BLUE_GREY_800),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        ),
        ft.Divider(),
        ft.ResponsiveRow(
            [
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("Live analysis", size=18, weight=ft.FontWeight.BOLD),
                            preview,
                            ft.Row(
                                [
                                    ft.Button("Select video", icon=ft.Icons.UPLOAD_FILE, on_click=choose_video),
                                    ft.Button("Start camera", icon=ft.Icons.VIDEOCAM,
                                              on_click=lambda e: setattr(status, "value", "Camera backend not connected yet") or page.update()),
                                ],
                                wrap=True,
                            ),
                            status,
                        ],
                        expand=True,
                    ),
                    col={"xs": 12, "md": 8},
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text("Current detection", size=18, weight=ft.FontWeight.BOLD),
                            action,
                            confidence,
                            ft.Divider(),
                            ft.Text("Pipeline status", weight=ft.FontWeight.BOLD),
                            ft.Text("UI ready · inference backend not connected"),
                            ft.Text("Next step: call your detector for each frame and update these controls."),
                        ],
                        spacing=16,
                    ),
                    padding=20,
                    border=ft.Border.all(1, ft.Colors.BLUE_GREY_700),
                    border_radius=12,
                    col={"xs": 12, "md": 4},
                ),
            ],
            spacing=16,
            run_spacing=16,
        ),
    )
