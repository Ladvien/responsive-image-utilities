import flet as ft
from responsive_image_utilities.image_labeler.label_manager import UnlabeledImagePair
from responsive_image_utilities.image_labeler.controls.image_with_label import (
    ImageWithLabel,
)


class ImagePairViewer(ft.Container):
    def __init__(self, pair: UnlabeledImagePair):
        super().__init__()

        # Create the image cards
        self.original = ImageWithLabel("Original", pair.original_image_path)
        self.noisy = ImageWithLabel("Noisy", pair.noisy_image_path)

        self.content = ft.Column(
            [
                ft.Row(
                    [self.original, self.noisy],
                    alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    expand=True,
                )
            ],
            spacing=20,
            expand=True,
        )

        self.bgcolor = "#1A1A2E"  # Very dark navy purple for contrast
        self.border_radius = 16
        self.padding = ft.Padding(20, 20, 20, 20)
        self.shadow = ft.BoxShadow(
            spread_radius=1,
            blur_radius=8,
            color="#00000080",  # subtle black shadow
            offset=ft.Offset(0, 4),
        )
        self.expand = True

    def update_images(self, pair: UnlabeledImagePair):
        self.original.update_images(pair.original_image_path)
        self.noisy.update_images(pair.noisy_image_path)
        self.update()
