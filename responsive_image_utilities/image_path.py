from __future__ import annotations
from dataclasses import dataclass
from PIL import Image as PILImage

from responsive_image_utilities.utils import ImageChecker


@dataclass
class ImagePath:
    path: str

    def load(self) -> PILImage:
        return PILImage.open(self.path)

    def is_valid_image(self) -> bool:
        return ImageChecker.is_valid_image(self.path)
