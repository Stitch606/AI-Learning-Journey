import flet as ft

def main(page: ft.Page):
    page.title = "تطبيق غبي جداً"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    
    # دالة بسيطة تتنفذ لما تضغط الزر
    def say_hello(e):
        text_output.value = "هيلو! 👋"
        page.update()

    # عنصر النص
    text_output = ft.Text(value="", size=24)
    
    # الزر البسيط
    btn = ft.ElevatedButton(text="اضغط هنا", on_click=say_hello)

    # إضافة العناصر للصفحة
    page.add(
        ft.Row([ft.Text("اضغط الزر بالأسفل:")], alignment=ft.MainAxisAlignment.CENTER),
        ft.Row([btn], alignment=ft.MainAxisAlignment.CENTER),
        ft.Row([text_output], alignment=ft.MainAxisAlignment.CENTER)
    )

ft.app(target=main)
