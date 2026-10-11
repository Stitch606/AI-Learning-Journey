import asyncio
import random

import flet as ft


GRID_SIZE = 16
TICK_SECONDS = 0.22
BG = "#101820"
BOARD_BG = "#1B2838"
SNAKE_COLOR = "#35D07F"
HEAD_COLOR = "#B7FF5A"
FOOD_COLOR = "#FF5C70"
TEXT_COLOR = "#F4F7FA"


def main(page: ft.Page):
    page.title = "Snake"
    page.padding = 12
    page.bgcolor = BG
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO

    snake = []
    direction = (1, 0)
    next_direction = (1, 0)
    food = (10, 8)
    score = 0
    running = False
    game_over = False

    score_text = ft.Text("النقاط: 0", size=20, color=TEXT_COLOR, weight=ft.FontWeight.BOLD)
    message = ft.Text("اضغط «ابدأ» للعب 🐍", size=14, color="#B9C5D3")
    cells = []
    board_rows = []

    for y in range(GRID_SIZE):
        row_cells = []
        row_controls = []
        for x in range(GRID_SIZE):
            cell = ft.Container(
                width=18,
                height=18,
                bgcolor=BOARD_BG,
                border_radius=4,
            )
            row_cells.append(cell)
            row_controls.append(cell)
        cells.append(row_cells)
        board_rows.append(ft.Row(controls=row_controls, spacing=2, alignment=ft.MainAxisAlignment.CENTER))

    board = ft.Container(
        content=ft.Column(controls=board_rows, spacing=2, tight=True),
        padding=8,
        bgcolor="#0B1118",
        border_radius=12,
    )

    def render():
        snake_set = set(snake)
        head = snake[0] if snake else None
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                pos = (x, y)
                if pos == head:
                    cells[y][x].bgcolor = HEAD_COLOR
                elif pos in snake_set:
                    cells[y][x].bgcolor = SNAKE_COLOR
                elif pos == food:
                    cells[y][x].bgcolor = FOOD_COLOR
                else:
                    cells[y][x].bgcolor = BOARD_BG
        score_text.value = f"النقاط: {score}"
        page.update()

    def spawn_food():
        empty = [
            (x, y)
            for y in range(GRID_SIZE)
            for x in range(GRID_SIZE)
            if (x, y) not in snake
        ]
        return random.choice(empty) if empty else None

    def reset_game(e=None):
        nonlocal snake, direction, next_direction, food, score, running, game_over
        snake = [(GRID_SIZE // 2, GRID_SIZE // 2),
                 (GRID_SIZE // 2 - 1, GRID_SIZE // 2),
                 (GRID_SIZE // 2 - 2, GRID_SIZE // 2)]
        direction = (1, 0)
        next_direction = (1, 0)
        food = spawn_food()
        score = 0
        running = True
        game_over = False
        message.value = "اجمع الطعام الأحمر وتجنب الحواف وجسمك!"
        start_button.text = "إعادة اللعب"
        render()

    def set_direction(dx, dy):
        nonlocal next_direction
        if (dx, dy) != (-direction[0], -direction[1]):
            next_direction = (dx, dy)

    def move_up(e):
        set_direction(0, -1)

    def move_down(e):
        set_direction(0, 1)

    def move_left(e):
        set_direction(-1, 0)

    def move_right(e):
        set_direction(1, 0)

    def make_button(label, handler):
        return ft.Button(
            content=ft.Text(label, size=22, weight=ft.FontWeight.BOLD),
            on_click=handler,
            width=76,
            height=54,
        )

    start_button = ft.Button(
        "ابدأ",
        on_click=reset_game,
        width=150,
        height=46,
    )

    controls = ft.Column(
        controls=[
            ft.Row([make_button("↑", move_up)], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row(
                [make_button("←", move_left), make_button("↓", move_down), make_button("→", move_right)],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=10,
            ),
        ],
        spacing=6,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    page.add(
        ft.Text("🐍 لعبة الثعبان", size=28, color=HEAD_COLOR, weight=ft.FontWeight.BOLD),
        score_text,
        board,
        controls,
        start_button,
        message,
        ft.Text("استخدم أزرار الاتجاهات للتحكم", size=12, color="#8190A0"),
    )

    async def game_loop():
        nonlocal direction, running, game_over, food, score
        while True:
            await asyncio.sleep(TICK_SECONDS)
            if not running or game_over:
                continue

            direction = next_direction
            hx, hy = snake[0]
            new_head = (hx + direction[0], hy + direction[1])

            if (
                new_head[0] < 0 or new_head[0] >= GRID_SIZE
                or new_head[1] < 0 or new_head[1] >= GRID_SIZE
                or new_head in snake[:-1]
            ):
                running = False
                game_over = True
                message.value = "انتهت اللعبة! اضغط «إعادة اللعب» للمحاولة مجدداً."
                page.update()
                continue

            snake.insert(0, new_head)
            if new_head == food:
                score += 1
                food = spawn_food()
                if food is None:
                    running = False
                    game_over = True
                    message.value = "فزت! ملأت اللوحة كلها 🎉"
            else:
                snake.pop()
            render()

    page.run_task(game_loop)


ft.run(main)
