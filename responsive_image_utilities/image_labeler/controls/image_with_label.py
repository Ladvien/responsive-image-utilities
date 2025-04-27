import flet as ft


class ImageWithLabel(ft.Column):
    def __init__(self, label_text: str, image: ft.Image):
        super().__init__()

        self.controls = [
            ft.Text(
                label_text,
                size=20,
                color=ft.colors.BLUE_900,
                text_align=ft.TextAlign.CENTER,
            ),
            image,
        ]
        self.alignment = ft.MainAxisAlignment.CENTER
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.expand = True
        self.animate_size = ft.Animation(
            duration=100, curve=ft.AnimationCurve.EASE_IN_OUT
        )
