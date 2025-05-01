import flet as ft
from responsive_image_utilities.image_utils.image_path import ImagePath


class ImageWithLabel(ft.Column):
    def __init__(
        self,
        label_text: str,
        image_path: ImagePath,
        color_scheme: ft.ColorScheme | None = None,
    ):
        super().__init__()
        self.image_path = image_path
        # Use provided theme or fallback
        self.color_scheme = color_scheme or ft.ColorScheme(
            primary="#7F00FF",
            on_primary="#FFFFFF",
            on_surface="#FFFFFF",
            on_background="#CCCCCC",
        )

        # Image control
        self.image = ft.Image(
            fit=ft.ImageFit.CONTAIN,
            expand=True,
            animate_size=ft.Animation(
                duration=300, curve=ft.AnimationCurve.EASE_IN_OUT
            ),
            border_radius=12,
        )

        self.image_container = ft.Container(
            content=self.image,
            bgcolor=self.color_scheme.surface_container,
            expand=True,
        )

        self.label_text = ft.Text(
            label_text,
            size=22,
            weight=ft.FontWeight.BOLD,
            color=self.color_scheme.on_surface,
            text_align=ft.TextAlign.CENTER,
        )

        self.name = ft.Text(
            image_path.name,
            size=14,
            color=self.color_scheme.on_background,
            text_align=ft.TextAlign.CENTER,
        )

        self.__set_images(image_path)

        # Assemble controls
        self.controls = [
            self.label_text,
            self.image_container,
            self.name,
        ]

        self.alignment = ft.MainAxisAlignment.CENTER
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.expand = True

    def __set_images(self, image_path: ImagePath):
        self.name.value = image_path.name
        self.image.src_base64 = image_path.load_as_base64()

    def update_images(self, image_path: ImagePath):
        """Update the displayed image from a new ImagePath."""
        self.__set_images(image_path)
        self.image.update()
        self.name.update()
