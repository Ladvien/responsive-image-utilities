import flet as ft
import os

from responsive_image_utilities.image_labeler import LabelAppFactory
from responsive_image_utilities.image_labeler import (
    LabelerConfig,
)

from responsive_image_utilities.image_labeler.label_manager_config import (
    LabelManagerConfig,
)
from responsive_image_utilities.image_utils import ImageLoader
from responsive_image_utilities.image_utils import ImageNoiser

# WILO: Set the path to your images directory
SOURCE_IMAGES_PATH = "/Users/ladvien/ladvien.com/content/images"
TRAIN_IMAGES_OUTPUT_PATH = "training_data/aiqa"

os.makedirs(TRAIN_IMAGES_OUTPUT_PATH, exist_ok=True)

# Create noisy images
# image_loader = ImageLoader(SOURCE_IMAGES_PATH, TRAIN_IMAGES_OUTPUT_PATH)
# source_images = image_loader.load_images()
# print(f"Found {len(source_images)} images.")

# Image noiser
# noiser = ImageNoiser()
# noiser.noise_images(
#     image_loader=image_loader,
#     output_folder=TRAIN_IMAGES_OUTPUT_PATH,
#     severity_range=(0.20, 0.95),
#     noise_functions=[
#         # ImageNoiser.add_gaussian_noise,
#         ImageNoiser.add_jpeg_compression,
#     ],
#     # samples=5000,
# )


config = LabelerConfig(
    title="Binary Image Labeler",
    window_width=800,
    window_height=700,
    window_resizable=True,
    theme_mode=ft.ThemeMode.DARK,
    label_manager_config=LabelManagerConfig(
        images_dir=SOURCE_IMAGES_PATH,
        output_dir=TRAIN_IMAGES_OUTPUT_PATH,
    ),
)


ft.app(target=LabelAppFactory.create_labeler_app(config))
