import flet as ft


class LabelingProgress(ft.Container):
    def __init__(
        self,
        value: float,
        instructions: ft.Control,
        progress_text: ft.Text,
    ):
        super().__init__()

        self.content = ft.Column(
            [
                instructions,
                ft.Row(
                    [progress_text],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.ProgressBar(
                    value=value,
                    width=300,
                    height=20,
                    bgcolor=ft.colors.BLUE_100,
                    color=ft.colors.BLUE_500,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.expand = 1  # Bottom 1/4
        self.padding = 20
