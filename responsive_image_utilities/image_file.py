from dataclasses import dataclass
from PIL import Image as PILImage, ImageStat

from responsive_image_utilities.image_path import ImagePath
from responsive_image_utilities.utils import ImageChecker


@dataclass
class ImageFile:
    potential_image: ImagePath
    image: PILImage.Image = None

    def __post_init__(self):
        if ImageChecker.is_valid_image(self.potential_image.path):
            self.image = PILImage.open(self.potential_image.path)
        else:
            raise Warning(f"Invalid image file: {self.potential_image}")
