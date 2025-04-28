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
    KeyboardBasedSlider,
)
from responsive_image_utilities.image_labeler.label_manager import LabelManager
from responsive_image_utilities.image_labeler.label_manager import UnlabeledImagePair


class ImageLabelerControl(ft.Column):
    def __init__(self, label_manager: LabelManager):
        super().__init__()

        self.label_manager = label_manager

        # Get initial image pair
        self.pair_to_label: UnlabeledImagePair = self.label_manager.get_unlabeled()

        # Controls
        self.image_pair_viewer = ImagePairViewer(self.pair_to_label)

        self.instructions = Instructions()
        self.progress_text = ft.Text()
        self.progress_bar = ft.ProgressBar(width=300)

        self.keyboard_based_slider = KeyboardBasedSlider()

        self.controls = [
            self.image_pair_viewer,
            LabelingProgress(self.instructions, self.progress_text, self.progress_bar),
            self.keyboard_based_slider,
        ]

        self.expand = True

        # --- Debounce Timer
        self._last_label_time = 0.0  # seconds since epoch
        self._debounce_interval = 0.5  # 0.5 seconds

        self.update_content()

    def update_content(self):
        """Update displayed images and progress."""
        self.pair_to_label = self.label_manager.get_unlabeled()

        if self.pair_to_label.original_image_path is None:
            print("No more images to label.")
            self.progress_text.value = "✅ All images labeled!"
            self.progress_bar.value = 1.0
            self.image_pair_viewer.original_image.src_base64 = None
            self.image_pair_viewer.noisy_image.src_base64 = None
        else:
            print(self.pair_to_label.original_image_path)
            self.image_pair_viewer.update_images(self.pair_to_label)
            # TODO: If you want progress tracking: uncomment and finish:
            # self.progress_text.value = f"{self.label_manager.current_index()}/{self.label_manager.image_count()} labeled"
            # self.progress_bar.value = self.label_manager.current_index() / self.label_manager.image_count()

    def __create_label(self, label: str):
        """Create and save a label for current image pair."""
        labeled_pair = self.pair_to_label.label(label)
        self.label_manager.save_label(labeled_pair)
        self.update_content()

    def _can_label(self) -> bool:
        """Debounce: check if enough time has passed since last label."""
        now = time.time()
        if now - self._last_label_time >= self._debounce_interval:
            self._last_label_time = now
            return True
        return False

    def handle_keyboard_event(self, key: Key | KeyCode) -> bool:
        """Handle keyboard events: slider and labeling."""
        if isinstance(key, KeyCode):
            return False

        if self.keyboard_based_slider.handle_keyboard_event(key):
            return True

        # Handle image labeling with debounce
        if key.name in ("right", "left"):
            if not self._can_label():
                return True  # Debounced: ignore too-fast presses

            if key.name == "right":
                self.__create_label("acceptable")
            elif key.name == "left":
                self.__create_label("unacceptable")

            return True

        return False
