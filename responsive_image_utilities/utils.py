from __future__ import annotations
from PIL import Image as PILImage


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
