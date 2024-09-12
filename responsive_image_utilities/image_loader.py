from glob import glob
from typing import List
from rich import print
from dataclasses import dataclass
from PIL import Image as PILImage, ImageStat

from responsive_image_utilities.constants import IMAGE_TYPES_LOWERCASE
from sewar.full_ref import uqi, sam, scc
import numpy as np


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


class ImagePath:
    path: str

    def __init__(self, path: str):
        self.path = path

    def load(self) -> PILImage.Image:
        return PILImage.open(self.path)

    def is_valid_image(self) -> bool:
        return ImageChecker.is_valid_image(self.path)

    def image_attributes(self):
        image = self.load()
        print(image.attributes)


@dataclass
class ImageFileData:
    potential_image: ImagePath
    image: PILImage.Image = None

    def __post_init__(self):
        if ImageChecker.is_valid_image(self.potential_image.path):
            self.image = PILImage.open(self.potential_image.path)
        else:
            raise Warning(f"Invalid image file: {self.potential_image}")


class ImageLoader:

    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder

        raw_image_paths = glob(f"{input_folder}/**/*", recursive=True)

        if raw_image_paths == [] or raw_image_paths is None:
            raise Exception(f"No files found in '{self.input_folder}'.")

        raw_image_paths.sort()

        self.image_paths = [ImagePath(path) for path in raw_image_paths]

    def load_images(self) -> List[ImageFileData]:
        return [
            ImageFileData(image_path)
            for image_path in self.image_paths
            if image_path.is_valid_image()
        ]
