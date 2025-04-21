import flet as ft

from responsive_image_utilities.image_labeler import (
    LabelAppFactory,
    BinaryLabelerPageConfig,
    ImageLoaderConfig,
)

# WILO: Set the path to your images directory
IMAGES_PATH = "path/to/your/images"

config = BinaryLabelerPageConfig(
    title="Binary Image Labeler",
    window_width=800,
    window_height=700,
    window_resizable=True,
    theme_mode=ft.ThemeMode.DARK,
    image_loader_config=ImageLoaderConfig(
        images_dir=IMAGES_PATH,
    ),
)


ft.app(target=LabelAppFactory.create_labeler_app(config))
