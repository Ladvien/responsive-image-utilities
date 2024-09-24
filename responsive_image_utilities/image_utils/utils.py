from PIL import Image as PILImage
import torch
import piq
import torchvision.transforms.functional as tf


class ImageChecker:

    @staticmethod
    def is_valid_image(path: str) -> bool:
        try:
            PILImage.open(path)
            return True
        except FileNotFoundError:
            print(f"File {path} not found. Maybe uppercase characters? Skipping...")
        except PILImage.UnidentifiedImageError:
            print(f"File {path} is not an image file. Skipping...")

        return False


def map_value(
    value: float,
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float,
) -> float:
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def pil_image_to_tensor(image: PILImage.Image) -> torch.Tensor:
    return tf.pil_to_tensor(image.convert("RGB")).permute(1, 2, 0)


def normalize_tensor_image(image: torch.Tensor) -> torch.Tensor:
    return image / 255.0


def preprocess_image(image: PILImage.Image) -> torch.Tensor:
    tensor = pil_image_to_tensor(image)
    normalized_tensor = normalize_tensor_image(tensor)
    if torch.cuda.is_available():
        normalized_tensor = normalized_tensor.cuda()
    return normalized_tensor
