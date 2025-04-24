import os
import csv
from random import uniform

from responsive_image_utilities.image_labeler.label_manager_config import (
    LabelManagerConfig,
)
from responsive_image_utilities.image_utils.image_loader import ImageLoader
from responsive_image_utilities.image_utils.image_path import ImagePath
from responsive_image_utilities.image_utils.image_noiser import ImageNoiser


class LabelManager:
    def __init__(self, config: LabelManagerConfig):
        self.config = config
        self.image_loader = ImageLoader(self.config.images_dir)

        self.labels = self._load_existing_labels()

        self.index = self._find_next_unlabeled_index()

    def _find_next_unlabeled_index(self) -> int:
        for i, image_path in enumerate(self.image_loader.image_paths):
            if image_path.path not in self.labels:
                return i
        return len(self.image_loader.image_paths)  # All done

    def _load_existing_labels(self) -> dict:
        labels = {}
        if os.path.exists(self.config.label_csv_path):
            with open(self.config.label_csv_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        labels[row[0]] = row[1]
        else:
            with open(self.config.label_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "label"])

        return labels

    def save_label(self, label: str):
        image_path = self.get_image_path()
        if image_path.path in self.labels:
            return  # Already labeled

        with open(self.config.label_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([image_path.path, label])

        self.labels[image_path.path] = label
        self.index += 1
        print(f"Labeled {image_path.name} as {label} — next index: {self.index}")

    def create_image_pair(self) -> tuple[ImagePath, ImagePath]:
        if self.index >= self.image_count():
            return None, None

        image_path = self.get_image_path()

        new_image = image_path.load()
        noisy_image_path = os.path.join(
            self.get_output_directory(),
            f"{image_path.name}_noisy.jpg",
        )
        new_image.save(noisy_image_path, quality=95)
        noisy_image_path = ImagePath(noisy_image_path)

        min_noise, max_noise = self.config.severity_range
        noise_level = uniform(min_noise, max_noise)
        noisy_image = ImageNoiser.add_jpeg_compression(
            new_image, noise_level, self.config.temporary_dir
        )
        noisy_image.save(noisy_image_path.path, quality=95)

        return image_path, noisy_image_path

    def get_image_path(self) -> ImagePath:
        return self.image_loader.image_paths[self.index]

    def image_count(self) -> int:
        return len(self.image_loader.image_paths)

    def get_labels(self) -> dict:
        return self.labels

    def get_output_directory(self) -> str:
        return self.config.output_dir

    def current_index(self) -> int:
        return self.index
