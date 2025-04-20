from typing import Optional
from PIL import Image
import torch
import clip
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PIL import Image as PILImage

from responsive_image_utilities.image_quality_assessor.train.basic import MLP


@dataclass
class ImageQualityAssessorConfig:
    model_load_folder: str
    model_load_name: str
    model_input_size: int
    clip_model_name: str = "ViT-L/14"
    device: str = "cpu"

    def __post_init__(self):
        path = Path(self.model_load_folder + "/")
        self.model_load_path = path / self.model_load_name


class ImageQualityAssessor:

    def __init__(self, config: ImageQualityAssessorConfig):
        self.config = config
        self.model = MLP(config.model_input_size)

        state = torch.load(config.model_load_path, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(config.device)
        self.model.eval()

        self.clip_model, self.preprocess = clip.load(
            config.clip_model_name, device=config.device
        )

    def score(
        self,
        *,
        pil_image: Optional[PILImage.Image] = None,
        image_path: Optional[str] = None
    ):
        if pil_image is None and image_path is None:
            raise ValueError("Either pil_image or image_path must be provided")

        if pil_image is None:
            pil_image = Image.open(image_path)

        image = self.preprocess(pil_image).unsqueeze(0).to(self.config.device)

        with torch.no_grad():
            # Assuming that `model2` is a pretrained CLIP model
            # and `image_features` is the output of `model2.encode_image(image)`
            image_features = self.clip_model.encode_image(image)
            image_features = image_features.float()
            image_features = image_features.to(self.config.device).cpu().numpy()
            image_features = self.normalize_features(image_features)

            prediction = (
                self.model(torch.tensor(image_features).to(self.config.device))
                .cpu()
                .numpy()[0][0]
            )

        return prediction

    def normalize_features(self, features, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(features, order, axis))
        l2[l2 == 0] = 1
        return features / np.expand_dims(l2, axis)
