from dataclasses import dataclass
from PIL import Image as PILImage, ImageStat
import numpy as np
from dataclasses import dataclass, asdict
import torch
import piq
from PIL import Image as PILImage
import torchvision.transforms.functional as tf


from responsive_image_utilities.image_path import ImagePath
from responsive_image_utilities.utils import ImageChecker


def map_value(
    value: float, in_min: float, in_max: float, out_min: float, out_max: float
) -> float:
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def pil_image_to_tensor(image: PILImage.Image) -> torch.Tensor:
    return tf.pil_to_tensor(image.convert("RGB")).permute(1, 2, 0)


def normalize_tensor_image(image: torch.Tensor) -> torch.Tensor:
    return image / 255.0


def calculate_brisque(image_tensor: torch.Tensor) -> float:
    image_tensor = torch.tensor(image_tensor).permute(2, 0, 1)[None, ...] / 255.0

    brisque_index: torch.Tensor = piq.brisque(
        image_tensor, data_range=1.0, reduction="none"
    )
    return brisque_index.item()


def preprocess_image(image: PILImage.Image) -> torch.Tensor:
    tensor = pil_image_to_tensor(image)
    normalized_tensor = normalize_tensor_image(tensor)
    if torch.cuda.is_available():
        normalized_tensor = normalized_tensor.cuda()
    return normalized_tensor


def calculate_clip_iqa_from_pil_image(image: PILImage.Image) -> float:
    image_tensor = preprocess_image(image)
    image_clip_iqa_index = piq.CLIPIQA(data_range=1.0).to(image_tensor)(
        image_tensor.clone().permute(2, 0, 1)[None, ...]
    )[0][0]

    return image_clip_iqa_index.item()


@dataclass
class ImageQuality:

    def __init__(self, image: PILImage.Image):
        self.image = image

    def create_highly_low_scoring_image(self) -> PILImage.Image:
        image_stat = ImageStat.Stat(self.image)
        mean = image_stat.mean
        std_dev = image_stat.stddev

        low_score_image = np.random.normal(mean, std_dev, self.image.size)
        return PILImage.fromarray(low_score_image)

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


class ImageFile:
    image_path: ImagePath
    quality: ImageQuality
    image: PILImage.Image = None

    def __init__(self, *, image_path: ImagePath = None, image: PILImage.Image = None):

        if image_path is not None:
            self.image_path = image_path
            self.image = PILImage.open(image_path.path)
        elif image is not None:
            self.image = image
        else:
            raise Exception("No image provided.")

        self.quality = ImageQuality(self.image)

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
