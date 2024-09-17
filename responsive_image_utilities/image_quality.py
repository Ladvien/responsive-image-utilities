from dataclasses import dataclass, asdict
from PIL import Image as PILImage, ImageStat
import torchvision.transforms.functional as tf


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

    def create_highly_distorted_images(self) -> PILImage.Image:
        distorted_image = self.image.copy()

        # Create a highly distorted image
        # Using each of the 17 distortion types

        # 1. Additive Gaussian noise
        distorted_image = ImageNoiser.add_noise(distorted_image, 1)
        # 2. Additive noise in color channels
        # 3. Spatially correlated noise
        # 4. Masked noise
        # 5. High frequency noise
        # 6. Impulse noise
        # 7. Quantization noise
        # 8. Gaussian blur
        # 9. Image denoising
        # 10 JPEG compression
        # 11. JPEG2000 compression
        # 12. JPEG transmission errors
        # 13. JPEG2000 transmission errors
        # 14. Non eccentricity pattern noise
        # 15. Local block-wise distortions of different intensity
        # 16. Mean shift (intensity shift)
        # 17. Contrast change

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
