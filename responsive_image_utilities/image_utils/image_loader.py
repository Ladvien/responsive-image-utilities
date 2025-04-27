from glob import glob
from pathlib import Path
from typing import Iterable, List
from PIL import Image as PILImage

from .image_path import ImagePath


class ImageLoader:

    def __init__(self, input_folder: str, extensions: List[str] = None):
        self.input_folder = input_folder

        if extensions:
            self.extensions = [ext.lower() for ext in extensions]
        else:
            self.extensions = [".jpg", ".jpeg", ".png"]

        raw_image_paths = list(Path(input_folder).rglob("*"))

        if not raw_image_paths:
            raise Exception(f"No files found in '{self.input_folder}'.")

        filtered_image_paths = [
            path for path in raw_image_paths if path.suffix.lower() in self.extensions
        ]

        if not filtered_image_paths:
            raise Exception(
                f"No files found in '{self.input_folder}' with extensions {self.extensions}."
            )

        filtered_image_paths.sort()

        potential_image_paths = [ImagePath(path) for path in filtered_image_paths]
        self.image_paths = [
            path for path in potential_image_paths if path.is_valid_image()
        ]

    def __iter__(self):
        """Allow iteration over loaded ImagePath objects."""
        return iter(self.image_paths)

    def __getitem__(self, index: int) -> ImagePath:
        """Allow index access to ImagePath objects."""
        return self.image_paths[index]

    def __len__(self) -> int:
        """Allow len(loader) to return number of images."""
        return len(self.image_paths)

    def iter_image_paths(self) -> Iterable[ImagePath]:
        return self.__iter__()

    def iter_images(self) -> Iterable[PILImage.Image]:
        """Yield all loaded PIL Images."""
        return (img_path.load() for img_path in self.image_paths)

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
