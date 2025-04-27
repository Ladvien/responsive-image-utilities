import flet as ft


class LabelingProgress(ft.Container):
    def __init__(
        self,
        instructions: ft.Control,
        progress_text: ft.Text,
        progress_bar: ft.ProgressBar,
    ):
        super().__init__()

        self.content = ft.Column(
            [
                instructions,
                ft.Row(
                    [progress_text],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                progress_bar,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.expand = 1  # Bottom 1/4
        self.padding = 20
