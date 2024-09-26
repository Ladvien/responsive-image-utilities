from responsive_image_utilities.image_quality_assessor import (
    IQATrainerConfig,
    IQATrainer,
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

IQATrainer(config)

from PIL import Image
import torch
import clip
from glob import glob
import numpy as np


images_folder = "tests/test_assets/images/data-warehouse/*.png"
images_paths = glob(images_folder)
image_path = images_paths[0]


def normalized(a, axis=-1, order=2):

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(MODEL_INPUT_SIZE)  # CLIP embedding dim is 768 for CLIP ViT L 14
state = torch.load("models/linear_predictor_l14_mse.pth")
model.load_state_dict(state)

device = "mps"

model.to(device)
model.eval()

# RN50x64 clip
# clip_name = "RN50x64"
clip_name = "ViT-L/14"
clip_model, preprocess = clip.load(clip_name, device=device)

# Show model info
print("Model info:")
print(preprocess)

good_image_path = "/Users/ladvien/responsive-image-utilities/tests/test_assets/images/data-warehouse/aws-admin-account-5.png"

print(preprocess(Image.open(good_image_path).convert("RGBA")).shape)


def predict_image_quality(
    image_path: str, model: torch.nn.Module, clip_model, preprocess
):
    pil_image = Image.open(image_path).convert("RGBA")

    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Assuming that `model2` is a pretrained CLIP model
        # and `image_features` is the output of `model2.encode_image(image)`
        image_features = clip_model.encode_image(image)
        image_features = image_features.float()
        # Use mps to convert the image features to numpy
        image_features = image_features.to(device).cpu().numpy()
        image_features = normalized(image_features)

        prediction = model(torch.tensor(image_features).to(device)).cpu().numpy()[0][0]

    print(
        f"Aesthetic score predicted by the model: {prediction} for image at path: {image_path}"
    )


good_image_path = "/Users/ladvien/responsive-image-utilities/tests/test_assets/images/data-warehouse/aws-admin-account-5.png"
bad_image_path = "/Users/ladvien/responsive-image-utilities/tests/test_assets/images/data-warehouse/aws-admin-account-bad-5.png"
bader_image_path = "/Users/ladvien/responsive-image-utilities/tests/test_assets/images/data-warehouse/aws-admin-account-bad-blur-5.png"

top_scoring_image = "/Users/ladvien/responsive-image-utilities/tests/test_assets/images/data-warehouse/Amazing-watercolor-paintings-4.jpg"

predict_image_quality(good_image_path, model, clip_model, preprocess)
predict_image_quality(bad_image_path, model, clip_model, preprocess)
predict_image_quality(bader_image_path, model, clip_model, preprocess)
predict_image_quality(top_scoring_image, model, clip_model, preprocess)
