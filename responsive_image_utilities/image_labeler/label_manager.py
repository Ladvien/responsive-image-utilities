from dataclasses import dataclass
import os
import csv
from random import uniform
import warnings
from rich import print
from pathlib import Path

from responsive_image_utilities.image_labeler.label_manager_config import (
    LabelManagerConfig,
)
from responsive_image_utilities.image_utils.image_loader import ImageLoader
from responsive_image_utilities.image_utils.image_path import ImagePath
from responsive_image_utilities.image_utils.image_noiser import ImageNoiser


@dataclass
class LabeledImagePair:
    original_image_path: ImagePath
    noisy_image_path: ImagePath
    label: str


@dataclass
class UnlabeledImagePair:
    original_image_path: ImagePath
    noisy_image_path: ImagePath

    def label(self, label: str) -> LabeledImagePair:
        if not isinstance(label, str):
            raise Exception(f"Label must be a string, received: {label}")

        return LabeledImagePair(self.original_image_path, self.noisy_image_path, label)


class LabelWriter:

    def __init__(self, path: str, overwrite: bool = False):
        self.path = Path(path)

        if not self.path.exists() or overwrite:
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["original_path", "noisy_path", "label"])

    def record_label(self, labeled_pair: LabeledImagePair):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    labeled_pair.original_image_path,
                    labeled_pair.noisy_image_path,
                    labeled_pair.label,
                ]
            )

    def is_labeled(self, image_path: str) -> bool:
        with open(self.path, "r", newline="") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                if row["original_path"] == image_path:
                    return True

        return False

    def get_labels(self) -> list[str]:
        with open(self.path, "r", newline="") as f:
            csv_reader = csv.DictReader(f)
            labels = [row["label"] for row in csv_reader]
        return labels


class LabelManager:
    def __init__(self, config: LabelManagerConfig):
        self.config = config
        self.image_loader = ImageLoader(self.config.images_dir)
        self.label_writer = LabelWriter(
            self.config.label_csv_path, self.config.overwrite_label_csv
        )
        self.labeled_image_paths = self.label_writer.get_labels()

    def save_label(self, labeled_pair: LabeledImagePair) -> None:
        if labeled_pair.original_image_path in self.labeled_image_paths:
            raise Exception(f"This image pair is already labeled. {labeled_pair}")

        self.label_writer.record_label(labeled_pair)
        self.labeled_image_paths.append(labeled_pair.original_image_path)

    def get_unlabeled(self) -> UnlabeledImagePair | None:
        image_path = next(self.image_loader)

        if image_path is None:
            return None

        new_image = image_path.load()
        noisy_image_path = os.path.join(
            self.config.output_dir,
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
        return UnlabeledImagePair(image_path, noisy_image_path)

    def unlabeled_count(self) -> int:
        return len(self.image_loader) - len(self.labeled_image_paths)

    def labeled_count(self) -> int:
        return len(self.labeled_image_paths)

    def percentage_complete(self) -> int:
        return self.labeled_count() / self.image_loader.total()

    def total(self) -> int:
        return self.image_loader.total()
