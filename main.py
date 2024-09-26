from responsive_image_utilities import ImageLoader
from responsive_image_utilities.image_quality_assessor.infer import (
    ImageQualityAssessor,
    ImageQualityAssessorConfig,
)

INPUT_FOLDER = "tests/test_assets/images"
OUTPUT_FOLDER = "tests/test_assets/output"

loader = ImageLoader(INPUT_FOLDER, OUTPUT_FOLDER)
image_paths = loader.load_images()

config = ImageQualityAssessorConfig(
    model_load_folder="models/",
    model_load_name="linear_predictor_l14_mse.pth",
    model_input_size=768,
)

iqa_assessor = ImageQualityAssessor(config)

for image_path in image_paths:
    print(
        f"Assessing image: {image_path.path} has score of {iqa_assessor.score_image(image_path.load())}"
    )
