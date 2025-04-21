from __future__ import annotations
from dataclasses import dataclass
from PIL import Image as PILImage
import os
import sys

from .utils import ImageChecker


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
        if os.path.isdir(self.path):
            return False

        return ImageChecker.is_valid_image(self.path)
