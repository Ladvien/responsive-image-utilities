import flet as ft


class Instructions(ft.Column):
    def __init__(self):
        super().__init__()

        self.instructions_label = ft.Text(
            "Keyboard Shortcuts:", size=20, color="#C792EA"
        )
        self.left_arrow_label = ft.Text("<- Left Arrow: Unacceptable", color="#C792EA")
        self.right_arrow_label = ft.Text("-> Right Arrow: Acceptable", color="#C792EA")

        self.controls = [
            ft.Column(
                [
                    self.instructions_label,
                    self.left_arrow_label,
                    self.right_arrow_label,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=1,
                tight=True,
                expand=True,
            )
        ]
