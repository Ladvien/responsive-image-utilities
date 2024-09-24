from __future__ import annotations
from dataclasses import dataclass
from PIL import Image as PILImage
import numpy as np

from responsive_image_utilities.utils import ImageChecker


@dataclass
class ImagePath:
    path: str

    def load(self, show_on_load: bool = False) -> PILImage:
        image = PILImage.open(self.path).convert("RGB")
        if show_on_load:
            print(f"Loading image from path: {self.path}")
            image.show()
        return image

    def is_valid_image(self) -> bool:
        return ImageChecker.is_valid_image(self.path)
