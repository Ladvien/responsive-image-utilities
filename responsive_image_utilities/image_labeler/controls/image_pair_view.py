import flet as ft

from responsive_image_utilities.image_labeler.controls.image_with_label import (
    ImageWithLabel,
)


class ImagePairViewer(ft.Container):
    def __init__(self, original_image: ft.Image, noisy_image: ft.Image):
        super().__init__()

        self.content = ft.Row(
            [
                ImageWithLabel("Original Image", original_image),
                ImageWithLabel("Noisy Image", noisy_image),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
            animate_size=ft.Animation(
                duration=300, curve=ft.AnimationCurve.EASE_IN_OUT
            ),
        )
        self.expand = 3  # Top 3/4
        self.padding = 20
