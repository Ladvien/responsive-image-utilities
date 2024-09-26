from responsive_image_utilities.image_quality_assessor import (
    IQATrainerConfig,
    IQATrainer,
)

from responsive_image_utilities.image_quality_assessor.infer import (
    ImageQualityAssessor,
    ImageQualityAssessorConfig,
)
from responsive_image_utilities.image_quality_assessor.train import MLP

MODEL_INPUT_SIZE = 768

config = IQATrainerConfig(
    training_data_path="training_data/ava_x_openai_clip_L14.npy",
    test_data_path="training_data/ava_y_openai_clip_L14.npy",
    model_save_folder="models/",
    model_save_name="linear_predictor_l14_mse.pth",
    model_input_size=MODEL_INPUT_SIZE,
    epochs=100,
)

# IQATrainer(config)

assessor = ImageQualityAssessor(
    ImageQualityAssessorConfig(
        model_load_folder="models",
        model_load_name="linear_predictor_l14_mse.pth",
        model_input_size=MODEL_INPUT_SIZE,
    )
)


good_image_path = "/Users/ladvien/responsive-image-utilities/tests/test_assets/images/data-warehouse/aws-admin-account-5.png"
bad_image_path = "/Users/ladvien/responsive-image-utilities/tests/test_assets/images/data-warehouse/aws-admin-account-bad-5.png"
bader_image_path = "/Users/ladvien/responsive-image-utilities/tests/test_assets/images/data-warehouse/aws-admin-account-bad-blur-5.png"

top_scoring_image = "/Users/ladvien/responsive-image-utilities/tests/test_assets/images/data-warehouse/Amazing-watercolor-paintings-4.jpg"

score = assessor.score_image(good_image_path)
print(f"Score: {score} for {good_image_path}")

score = assessor.score_image(bad_image_path)
print(f"Score: {score} for {bad_image_path}")

score = assessor.score_image(bader_image_path)
print(f"Score: {score} for {bader_image_path}")

score = assessor.score_image(top_scoring_image)
print(f"Score: {score} for {top_scoring_image}")
