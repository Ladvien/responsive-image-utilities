import flet as ft
from responsive_image_utilities.image_labeler.label_manager import UnlabeledImagePair
from responsive_image_utilities.image_labeler.controls.image_with_label import (
    ImageWithLabel,
)


class ImagePairViewer(ft.Container):
    def __init__(self, pair: UnlabeledImagePair):
        super().__init__()
        self.original = ImageWithLabel("Original Image", pair.original_image_path)
        self.noisy = ImageWithLabel("Noisy Image", pair.noisy_image_path)

        self.content = ft.Row(
            [
                self.original,
                self.noisy,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
            animate_size=ft.Animation(
                duration=300, curve=ft.AnimationCurve.EASE_IN_OUT
            ),
        )
        self.expand = 3
        self.padding = 20

    def update_images(self, pair: UnlabeledImagePair):
        self.original.update_images(pair.original_image_path)
        self.noisy.update_images(pair.noisy_image_path)
        self.update()
