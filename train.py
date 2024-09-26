from glob import glob

from responsive_image_utilities.image_quality_assessor import (
    IQATrainerConfig,
    ImageQualityAssessmentTrainer,
)

from responsive_image_utilities.image_quality_assessor.infer import (
    ImageQualityAssessor,
    ImageQualityAssessorConfig,
)

MODEL_INPUT_SIZE = 768

config = IQATrainerConfig(
    training_data_path="training_data/ava_x_openai_clip_L14.npy",
    test_data_path="training_data/ava_y_openai_clip_L14.npy",
    model_save_folder="models/",
    model_save_name="linear_predictor_l14_mse.pth",
    model_input_size=MODEL_INPUT_SIZE,
    epochs=30,
)

# ImageQualityAssessmentTrainer(config)

assessor = ImageQualityAssessor(
    ImageQualityAssessorConfig(
        model_load_folder="models",
        model_load_name="linear_predictor_l14_mse.pth",
        model_input_size=MODEL_INPUT_SIZE,
    )
)


image_paths = glob("tests/test_assets/images/**/*.png", recursive=True)

for image_path in image_paths:
    score = assessor.score_image(image_path)
    print(f"Score: {score} for {image_path}")
