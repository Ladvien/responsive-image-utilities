from responsive_image_utilities.image_quality_assessor.train.dynamic import (
    IQAConfig,
    ImageQualityClassifierTrainer,
)


if __name__ == "__main__":
    config = IQAConfig(
        csv_path="training_data/aiqa/labels.csv",
        model_save_folder="models",
        model_save_name="mobilenetv2_binary_iqa.pth",
        batch_size=32,
        epochs=10,
    )

    trainer = ImageQualityClassifierTrainer(config)
    trainer.train()
