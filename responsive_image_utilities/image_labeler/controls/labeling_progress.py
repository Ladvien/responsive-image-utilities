import flet as ft


class LabelingProgress(ft.Container):
    def __init__(
        self,
        value: float,
        progress_text: ft.Text,
        expand: int = 1,
    ):
        super().__init__()

        self.content = ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        f"{progress_text}",
                        size=14,
                        weight=ft.FontWeight.BOLD,
                        color=ft.colors.WHITE,
                        text_align=ft.TextAlign.RIGHT,
                    ),
                    ft.ProgressBar(
                        value=value,
                        bgcolor="#1E1E2F",
                        color="#C792EA",
                        height=12,
                        expand=True,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    def update_progress(self, value: float, progress_text: str):
        self.content.content.controls[0].value = progress_text
        self.content.content.controls[1].value = value
        self.update()
