from glob import glob
from typing import List
from sewar.full_ref import uqi, sam, scc
import numpy as np


from responsive_image_utilities.image_file import ImageFile
from responsive_image_utilities.image_path import ImagePath
from responsive_image_utilities.utils import ImageChecker


class ImageLoader:

    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder

        raw_image_paths = glob(f"{input_folder}/**/*", recursive=True)

        if raw_image_paths == [] or raw_image_paths is None:
            raise Exception(f"No files found in '{self.input_folder}'.")

        raw_image_paths.sort()

        self.image_paths = [ImagePath(path) for path in raw_image_paths]

    def load_images(self) -> List[ImageFile]:
        return [
            ImageFile(image_path=image_path)
            for image_path in self.image_paths
            if image_path.is_valid_image()
        ]