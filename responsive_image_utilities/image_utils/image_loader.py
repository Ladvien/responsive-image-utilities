from glob import glob
from typing import List
from PIL import Image as PILImage

from .image_path import ImagePath


class ImageLoader:

    def __init__(self, input_folder: str, extensions: List[str] = None):
        self.input_folder = input_folder

        self.extensions = extensions if extensions else [".jpg", ".jpeg", ".png"]

        raw_image_paths = glob(f"{input_folder}/**/*", recursive=True)

        if raw_image_paths == [] or raw_image_paths is None:
            raise Exception(f"No files found in '{self.input_folder}'.")

        raw_image_paths.sort()

        filtered_image_paths = [
            path for path in raw_image_paths if path.endswith(tuple(self.extensions))
        ]

        if filtered_image_paths == [] or filtered_image_paths is None:
            raise Exception(
                f"No files found in '{self.input_folder}' with extensions {self.extensions}."
            )

        if len(filtered_image_paths) == 0:
            raise Exception(
                f"No files found in '{self.input_folder}' with extensions {self.extensions}."
            )

        self.image_paths = [ImagePath(path) for path in filtered_image_paths]

    def load_images(self) -> List[ImagePath]:
        return [
            image_path for image_path in self.image_paths if image_path.is_valid_image()
        ]

    def get_all_images(self) -> List[PILImage.Image]:
        return [image_path.load() for image_path in self.image_paths]

    def get_all_image_paths(self) -> List[str]:
        return [image_path.path for image_path in self.image_paths]

    def get_image(self, index: int) -> PILImage.Image:
        if index < 0 or index >= len(self.image_paths):
            raise IndexError("Index out of range.")
        return self.image_paths[index].load()

    def get_image_path(self, index: int) -> ImagePath:
        return self.image_paths[index]
