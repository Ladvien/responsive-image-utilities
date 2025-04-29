import time
from typing import Callable
import flet as ft
from rich import print
from pynput.keyboard import Key, KeyCode

from responsive_image_utilities.image_labeler.controls.image_pair_view import (
    ImagePairViewer,
)
from responsive_image_utilities.image_labeler.controls.instructions import Instructions
from responsive_image_utilities.image_labeler.controls.labeling_progress import (
    LabelingProgress,
)
from responsive_image_utilities.image_labeler.controls.noise_slider import (
    PersistentLabeledRangeSlider,
)
from responsive_image_utilities.image_labeler.label_manager import LabelManager
from responsive_image_utilities.image_labeler.label_manager import UnlabeledImagePair


class ImageLabelerControl(ft.Column):
    def __init__(self, label_manager: LabelManager):
        super().__init__()

        self.label_manager = label_manager
        self.unlabeled_pair = self.label_manager.new_unlabeled()
        self.image_pair_viewer = ImagePairViewer(self.unlabeled_pair)

        def on_slider_update(
            event: ft.ControlEvent, start_value: float, end_value: float
        ):
            """Update the noise slider value."""
            self.label_manager.set_severity_level(start_value, end_value)
            self.unlabeled_pair = self.label_manager.update_severity(
                self.unlabeled_pair
            )
            self.image_pair_viewer.update_images(self.unlabeled_pair)

        self.noise_slider = PersistentLabeledRangeSlider(on_end_change=on_slider_update)

        self.controls = [
            self.image_pair_viewer,
            self.noise_slider,
            LabelingProgress(
                self.label_manager.percentage_complete(),
                Instructions(),
                ft.Text(
                    f"{self.label_manager.labeled_count()}/{self.label_manager.total()} labeled"
                ),
            ),
        ]

        self.expand = True

        # --- Debounce Timer
        self._last_label_time = 0.0  # seconds since epoch
        self._debounce_interval = 0.5  # 0.5 seconds

    def on_mount(self):
        self.update_content()

    def update_content(self) -> None:
        """Update displayed images and progress."""
        self.unlabeled_pair = self.label_manager.new_unlabeled()
        self.image_pair_viewer.update_images(self.unlabeled_pair)

    def __label_image(self, label: str) -> None:
        labeled_pair = self.unlabeled_pair.label(label)
        self.label_manager.save_label(labeled_pair)
        self.update_content()

    def __can_label(self) -> bool:
        now = time.time()
        if now - self._last_label_time >= self._debounce_interval:
            self._last_label_time = now
            return True
        return False

    def handle_keyboard_event(self, key: Key | KeyCode) -> bool:
        """Handle keyboard events: slider and labeling."""

        # NOTE: This method only wants key.
        if not isinstance(key, Key):
            return False

        # Handle image labeling with debounce
        if key.name in ("right", "left"):
            if not self.__can_label():
                return True

            if key.name == "right":
                self.__label_image("acceptable")
            elif key.name == "left":
                self.__label_image("unacceptable")

            return True

        return False
