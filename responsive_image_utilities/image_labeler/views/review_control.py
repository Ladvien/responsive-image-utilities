import flet as ft
from pynput.keyboard import Key, KeyCode
import time

from responsive_image_utilities.image_labeler.controls.image_pair_view import (
    ImagePairViewer,
)
from responsive_image_utilities.image_labeler.label_manager import (
    LabeledImagePair,
)


class ReviewControl(ft.Column):
    def __init__(
        self,
        labeled_image_pairs: list[LabeledImagePair],
        color_scheme: ft.ColorScheme | None = None,
    ):
        super().__init__()
        self.labeled_image_pairs = labeled_image_pairs
        self.__index = 0
        self.image_pair_viewer = ImagePairViewer(
            self.labeled_image_pairs[self.__index],
            color_scheme,
        )

        self.color_scheme = color_scheme or ft.ColorScheme()

        self.__last_key_press_time = 0
        self.__debounce_interval = 0.2  # seconds

        self.expand = True
        self.alignment = ft.MainAxisAlignment.START
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.spacing = 0

        self.controls = [
            ft.Container(
                content=self.image_pair_viewer,
                bgcolor=self.color_scheme.primary,
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
                                    # ft.Container(
                                    #     content=Instructions(
                                    #         color_scheme=self.color_scheme
                                    #     ),
                                    #     padding=10,
                                    #     expand=True,
                                    #     alignment=ft.alignment.center_left,
                                    # ),
                                    # ft.Container(
                                    #     content=self.noise_control,
                                    #     padding=10,
                                    #     bgcolor=self.color_scheme.primary,
                                    #     border_radius=10,
                                    #     expand=True,
                                    #     alignment=ft.alignment.center_right,
                                    # ),
                                    # ft.Container(
                                    #     content=self.progress_area,
                                    #     padding=10,
                                    #     bgcolor=self.color_scheme.primary,
                                    #     border_radius=10,
                                    #     expand=True,
                                    #     alignment=ft.alignment.center_right,
                                    # ),
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            expand=3,  # 3/4 width
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

    def update_image_pair_viewer(self):
        print(f"Updating image pair viewer with index: {self.__index}")
        labeled_pair = self.labeled_image_pairs[self.__index]
        self.image_pair_viewer.update_images(labeled_pair)

    def handle_keyboard_event(self, key: Key | KeyCode) -> bool:
        if not isinstance(key, Key):
            return False

        if not self.__debounce_keypress():
            return False

        if key.name in ("right", "left"):
            if key.name == "right":
                # if self.__index < len(self.labeled_image_pairs) - 1:
                self.__index += 1
            elif key.name == "left":
                # if self.__index > 0:
                self.__index -= 1

            self.update_image_pair_viewer()
            self.update()

            return True

        elif key.name == "space" and self.__can_label():
            self.__resample_images()
            return True

    def __debounce_keypress(self) -> bool:
        now = time.time()
        if now - self.__last_key_press_time >= self.__debounce_interval:
            self.__last_key_press_time = now
            return True
        return False
