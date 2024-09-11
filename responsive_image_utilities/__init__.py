from glob import glob
from typing import List
from rich import print
from dataclasses import dataclass
from PIL import Image as PILImage, ImageStat

from src.constants import IMAGE_TYPES
from sewar.full_ref import uqi, sam, scc
import numpy as np


class ImagePath:
    path: str

    def __init__(self, path: str):
        self.path = path

    def load(self) -> PILImage.Image:
        # TODO: Add exception handling
        return PILImage.open(self.path)

    def is_image_safe_to_resize(self):
        raise NotImplementedError

    def image_attributes(self):
        image = self.load()
        print(image.attributes)


@dataclass
class ImageData:
    image: PILImage.Image
    path: str


class ImageFolder:

    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder

        self.raw_image_paths = glob(f"{input_folder}/**/*", recursive=True)
        self.raw_image_paths.sort()

        # TODO: What if file does not contain an extension
        # TODO: What if file does not contain '.' in the name
        self.raw_image_paths = [
            path for path in self.raw_image_paths if path.split(".")[-1] in IMAGE_TYPES
        ]

        self.image_paths = [ImagePath(path) for path in self.raw_image_paths]

    def load_images(self) -> List[ImageData]:
        return [ImageData(image.load(), image.path) for image in self.image_paths]

    def image_stats(self):
        data = self.load_images()
        for datum in data:
            resize_tuple = datum.image.size[0] // 3, datum.image.size[1] // 3
            low_res = datum.image.copy().resize(resize_tuple)
            low_res = np.array(low_res.resize(datum.image.size))
            image = np.array(datum.image)

            # Universal Quality Index should be high when
            # the original image and the low resolution image
            # are similar
            uqi_score = uqi(image, low_res)

            # Structural Similarity Index should be low when
            # the original image and the low resolution image
            # are different
            sam_score = sam(image, low_res)

            # Structural Content Correlation should be high when
            # the original image and the low resolution image
            # are similar
            scc_score = scc(image, low_res)

            print(
                f"UQI Score: {uqi_score}, SAM: {sam_score}, SCC: {scc_score} for image {datum.path}"
            )
