import flet as ft

from responsive_image_utilities.image_utils.image_path import ImagePath


class ImageWithLabel(ft.Column):
    def __init__(self, label_text: str, image_path: ImagePath):
        super().__init__()
        # Create Image controls inside here
        self.image = ft.Image(
            fit=ft.ImageFit.CONTAIN,
            expand=True,
            animate_size=ft.Animation(
                duration=300, curve=ft.AnimationCurve.EASE_IN_OUT
            ),
        )

        self.controls = [
            ft.Text(
                label_text,
                size=20,
                color=ft.colors.BLUE_900,
                text_align=ft.TextAlign.CENTER,
            ),
            self.image,
            ft.Text(
                image_path.name,
                size=12,
                color=ft.colors.BLUE_900,
                text_align=ft.TextAlign.CENTER,
            ),
        ]
        self.alignment = ft.MainAxisAlignment.CENTER
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.expand = True
        self.animate_size = ft.Animation(
            duration=100, curve=ft.AnimationCurve.EASE_IN_OUT
        )

        # Now load the initial pair
        self.update_images(image_path)

    def update_images(self, image_path: ImagePath):
        """Update the displayed images from a new UnlabeledImagePair."""
        self.image.src_base64 = image_path.load_as_base64()
