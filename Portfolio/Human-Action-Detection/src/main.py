import logging

import flet as ft

from views.dashboard import build_dashboard

LOG_LEVEL = logging.INFO
logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("app").setLevel(LOG_LEVEL)


def main(page: ft.Page):
    page.title = "Human Action Detection"
    page.theme_mode = ft.ThemeMode.DARK
    build_dashboard(page)


ft.run(main)
