import flet as ft


def main(page: ft.Page):
    page.title = "تطبيق غبي جداً"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    def say_hello(e):
        text_output.value = "هيلو! 👋"
        page.update()

    text_output = ft.Text(value="", size=24)
    btn = ft.ElevatedButton(text="اضغط هنا", on_click=say_hello)

    page.add(
        ft.Row([ft.Text("اضغط الزر بالأسفل:")], alignment=ft.MainAxisAlignment.CENTER),
        ft.Row([btn], alignment=ft.MainAxisAlignment.CENTER),
        ft.Row([text_output], alignment=ft.MainAxisAlignment.CENTER),
    )


ft.run(main)
