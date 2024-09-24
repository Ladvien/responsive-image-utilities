from __future__ import annotations  # Needed for type hinting
from dataclasses import dataclass, asdict
from PIL import Image as PILImage

from responsive_image_utilities.image_noiser import ImageNoiser
from responsive_image_utilities.utils import (
    calculate_brisque,
    calculate_clip_iqa_from_pil_image,
    preprocess_image,
)


@dataclass
class ImageQuality:

    def __init__(self, image: PILImage.Image):
        self.image = image
        self.distorted_image = None

    def create_highly_distorted_images(self, severity: int = 3) -> None:
        self.distorted_image = ImageNoiser.with_noise(self.image, severity)

    def brisque(self, other_image: PILImage.Image) -> float:
        image_tensor = preprocess_image(self.image)
        other_image_tensor = preprocess_image(other_image)

        image_brisque_index = calculate_brisque(image_tensor)
        other_image_brisque_index = calculate_brisque(other_image_tensor)

        return {
            "image_brisque_index": image_brisque_index,
            "other_image_brisque_index": other_image_brisque_index,
        }

    def calculate_clip_iqa(self, other_image: PILImage.Image) -> float:
        return calculate_clip_iqa_from_pil_image(other_image)

    def __dict__(self) -> dict[str, float]:
        return asdict(self)
