from adaptive_labeler import LabelerConfig, LabelManagerConfig, LabelAppFactory
import flet as ft
import os


# WILO: Set the path to your images directory
SOURCE_IMAGES_PATH = "/Users/ladvien/ladvien.com/content/images"
TRAIN_IMAGES_OUTPUT_PATH = "training_data/aiqa"
TEMPORARY_IMAGES_OUTPUT_PATH = f"{TRAIN_IMAGES_OUTPUT_PATH}/temporary"

NUM_IMAGE_SAMPLES = 5000

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
        temporary_dir=TEMPORARY_IMAGES_OUTPUT_PATH,
        image_samples=NUM_IMAGE_SAMPLES,
    ),
)


ft.app(target=LabelAppFactory.create_labeler_app(config))
