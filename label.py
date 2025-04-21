import flet as ft
import os
from responsive_image_utilities.image_labeler import (
    LabelAppFactory,
    BinaryLabelerPageConfig,
)
from responsive_image_utilities.image_utils import ImageLoader

# WILO: Set the path to your images directory
SOURCE_IMAGES_PATH = "/Users/ladvien/ladvien.com/content/images"
TRAIN_IMAGES_OUTPUT_PATH = "training_data/aiqa"

os.makedirs(TRAIN_IMAGES_OUTPUT_PATH, exist_ok=True)

# Create noisy images
image_loader = ImageLoader(SOURCE_IMAGES_PATH, TRAIN_IMAGES_OUTPUT_PATH)
source_images = image_loader.load_images()
print(f"Found {len(source_images)} images.")

quit()

config = BinaryLabelerPageConfig(
    title="Binary Image Labeler",
    window_width=800,
    window_height=700,
    window_resizable=True,
    theme_mode=ft.ThemeMode.DARK,
    image_loader_config=LabelerImageLoaderConfig(
        images_dir=TRAIN_IMAGES_OUTPUT_PATH,
    ),
)


ft.app(target=LabelAppFactory.create_labeler_app(config))
