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

        self.expand = True
        self.alignment = ft.MainAxisAlignment.START
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.spacing = 0

        def on_slider_update(
            event: ft.ControlEvent, start_value: float, end_value: float
        ):
            self.label_manager.set_severity_level(start_value, end_value)
            self.unlabeled_pair = self.label_manager.update_severity(
                self.unlabeled_pair
            )
            self.image_pair_viewer.update_images(self.unlabeled_pair)

        self.noise_slider = PersistentLabeledRangeSlider(on_end_change=on_slider_update)

        self.controls = [
            ft.Container(
                content=self.image_pair_viewer,
                bgcolor="#1E1E2F",
                padding=20,
                border_radius=12,
                expand=3,  # Top: 3/4 screen
            ),
            ft.Container(
                content=ft.Row(
                    [
                        # 3/4 of width: Instructions + Slider
                        ft.Container(
                            content=ft.Row(
                                [
                                    ft.Container(
                                        content=Instructions(),
                                        padding=10,
                                        expand=True,
                                        alignment=ft.alignment.center_left,
                                    ),
                                    ft.Container(
                                        content=self.noise_slider,
                                        padding=10,
                                        bgcolor="#1A1A2E",
                                        border_radius=10,
                                        expand=True,
                                        alignment=ft.alignment.center_right,
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            expand=3,  # 3/4 width
                        ),
                        # 1/4 of width: Progress Bar
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text(
                                        f"{self.label_manager.labeled_count()}/{self.label_manager.total()} labeled",
                                        size=14,
                                        weight=ft.FontWeight.BOLD,
                                        color=ft.colors.WHITE,
                                        text_align=ft.TextAlign.RIGHT,
                                    ),
                                    ft.ProgressBar(
                                        value=self.label_manager.percentage_complete(),
                                        bgcolor="#555",
                                        color="#C792EA",
                                        height=12,
                                        expand=True,
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            expand=1,  # 1/4 width
                            padding=10,
                            bgcolor="#2A2A40",
                            border_radius=10,
                        ),
                    ],
                    expand=True,
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=20,
                expand=1,  # Bottom: 1/4 screen
            ),
        ]

        # --- Debounce Timer
        self._last_label_time = 0.0
        self._debounce_interval = 0.5

    def on_mount(self):
        self.update_content()

    def update_content(self) -> None:
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
        if not isinstance(key, Key):
            return False
        if key.name in ("right", "left") and self.__can_label():
            self.__label_image("acceptable" if key.name == "right" else "unacceptable")
            return True
        return False
