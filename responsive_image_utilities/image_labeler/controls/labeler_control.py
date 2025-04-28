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


class ImageLabelerControl(ft.Column):
    def __init__(self, label_manager: LabelManager):

        super().__init__()

        self.label_manager = label_manager

        self.instructions = Instructions()
        self.progress_text = ft.Text()
        self.progress_bar = ft.ProgressBar(width=300)

        self.original = ft.Image(
            fit=ft.ImageFit.CONTAIN,
            expand=True,
            animate_size=ft.Animation(
                duration=300, curve=ft.AnimationCurve.EASE_IN_OUT
            ),
        )
        self.noisy = ft.Image(
            fit=ft.ImageFit.CONTAIN,
            expand=True,
            animate_size=ft.Animation(
                duration=300, curve=ft.AnimationCurve.EASE_IN_OUT
            ),
        )

        self.keyboard_based_slider = KeyboardBasedSlider()
        # self.slider = ft.Slider(min=0.0, max=1.0)

        self.controls = [
            ImagePairViewer(self.original, self.noisy),
            LabelingProgress(self.instructions, self.progress_text, self.progress_bar),
            self.keyboard_based_slider,
            # self.slider,
        ]

        self.pair_to_label = self.label_manager.get_unlabeled()

        self.expand = True
        self.update_content()

    def update_content(self):
        self.pair_to_label = self.label_manager.get_unlabeled()

        if self.pair_to_label.original_image_path is None:
            print("No more images to label.")
            self.progress_text.value = "✅ All images labeled!"
            self.progress_bar.value = 1.0
            self.original.src_base64 = None
            self.noisy.src_base64 = None
        else:
            print(self.pair_to_label.original_image_path)
            self.original.src_base64 = (
                self.pair_to_label.original_image_path.load_as_base64()
            )
            self.noisy.src_base64 = self.pair_to_label.noisy_image_path.load_as_base64()
            # self.progress_text.value = f"{self.label_manager.c()}/{self.label_manager.image_count()} labeled"
            # self.progress_bar.value = (
            #     self.label_manager.current_index() / self.label_manager.image_count()
            # )

    def handle_keyboard_event(self, key: Key | KeyCode):
        if isinstance(key, KeyCode):
            return False

        if self.keyboard_based_slider.handle_keyboard_event(key):
            return True

        if key.name not in ["right", "left"]:
            return False

        label = ""
        if key.name == "right":
            print("Arrow Right")
            label = "acceptable"
        if key.name == "left":
            print("Arrow Left")
            label = "unacceptable"

        labeled_pair = self.pair_to_label.label(label)
        self.label_manager.save_label(labeled_pair)
        self.update_content()

        return True
