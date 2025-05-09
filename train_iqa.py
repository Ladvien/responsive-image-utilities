from responsive_image_utilities.image_quality_assessor.train.dynamic import (
    IQAConfig,
    ImageQualityClassifierTrainer,
)


if __name__ == "__main__":
    config = IQAConfig(
        csv_path="training_data/aiqa/labels.csv",
        model_save_folder="models",
        model_save_name="siamese_resnet_binary_iqa.pth",
        batch_size=32,
        epochs=500,
        early_stopping_patience=50,
        test_split=0.2,
    )

    trainer = ImageQualityClassifierTrainer(config)
    trainer.train()
