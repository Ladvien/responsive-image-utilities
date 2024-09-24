from __future__ import annotations  # Needed for type hinting

from PIL import Image as PILImage
import numpy as np
from PIL import Image as PILImage
import torchvision.transforms.functional as tf

from responsive_image_utilities.image_path import ImagePath
from responsive_image_utilities.image_quality import ImageQuality


class ImageFile:
    image_path: ImagePath
    quality: ImageQuality
    image: PILImage.Image = None

    def __init__(
        self,
        *,
        image_path: ImagePath = None,
        image: PILImage.Image = None,
        show_on_load: bool = False,
    ):

        if image_path is not None:
            self.image_path = image_path
            self.image = self.image_path.load(show_on_load)
        elif image is not None:
            self.image = image
        else:
            raise Exception("No image provided.")

        self.quality = ImageQuality(self.image)

    def as_numpy_array(self) -> np.ndarray:
        return np.array(self.image)

    def resize(self, size: tuple[int, int]) -> "ImageFile":
        return ImageFile(image=self.image.resize(size))

    def resize_by_width(self, new_width: int) -> "ImageFile":
        if new_width >= self.image.size[0]:
            raise Exception("New width must be smaller than the current width.")

        width_percent = new_width / self.image.size[0]
        new_height = int(self.image.size[1] * width_percent)
        image = self.resize((new_width, new_height))
        return ImageFile(image=image.image)

    def test_resize_quality(self, width: int) -> float:
        resized_image = self.resize_by_width(width)
        original_brisque_score = self.quality.brisque(self.image)
        resized_brisque_score = self.quality.brisque(resized_image.image)

        print(f"Original shape: {self.image.size}")
        print(f"Resized shape: {resized_image.image.size}")

        original_clip_iqa_score = self.quality.calculate_clip_iqa(self.image)
        noisy_clip_iqa_score = self.quality.calculate_clip_iqa(
            self.quality.create_highly_low_scoring_image()
        )
        resized_clip_iqa_score = self.quality.calculate_clip_iqa(resized_image.image)

        return {
            "original_brisque_score": original_brisque_score,
            "resized_brisque_score": resized_brisque_score,
            "original_clip_iqa_score": original_clip_iqa_score,
            "noisy_clip_iqa_score": noisy_clip_iqa_score,
            "resized_clip_iqa_score": resized_clip_iqa_score,
        }

    def __str__(self) -> str:
        try:
            self.image_path
            return f"ImageFile(image_path={self.image_path})"
        except AttributeError:
            image_size = self.image.size
            return f"ImageFile(image_path=None, Image=({image_size}))"
