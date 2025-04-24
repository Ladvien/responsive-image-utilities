import flet as ft
import os
from random import uniform


from responsive_image_utilities.image_labeler.label_manager import LabelManager
from responsive_image_utilities.image_utils.image_noiser import ImageNoiser
from responsive_image_utilities.image_utils.image_path import ImagePath


class ImageLabelerControl(ft.Row):
    def __init__(self, label_manager: LabelManager):
        super().__init__()
        self.label_manager = label_manager

        self.original = ft.Image(width=700, height=500, fit=ft.ImageFit.CONTAIN)
        self.noisy = ft.Image(width=700, height=500, fit=ft.ImageFit.CONTAIN)

        self.shortcut_info = ft.Text(
            "Keyboard Shortcuts:\n <- Left Arrow: Unacceptable\n -> Right Arrow: Acceptable",
            size=14,
            color=ft.Colors.GREY_600,
            text_align=ft.TextAlign.CENTER,
        )

        self.progress_text = ft.Text()
        self.progress_bar = ft.ProgressBar(width=300)

        self.controls = [
            ft.Column(
                [
                    ft.Row(
                        [
                            self.original,
                            self.noisy,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    self.shortcut_info,
                    ft.Row([self.progress_text], alignment=ft.MainAxisAlignment.CENTER),
                    self.progress_bar,
                ],
                expand_loose=True,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
        ]

        self.update_content()

    def before_update(self):
        self.update_content()

    def update_content(self):
        image_path, noisy_image_path = self.label_manager.create_image_pair()
        if image_path is None:
            print("No more images to label.")
            self.progress_text.value = "✅ All images labeled!"
            self.progress_bar.value = 1.0
            self.original.src_base64 = None
            self.noisy.src_base64 = None
        else:
            self.original.src_base64 = image_path.load_as_base64()
            self.noisy.src_base64 = noisy_image_path.load_as_base64()
            self.progress_text.value = f"{self.label_manager.current_index()}/{self.label_manager.image_count()} labeled"
            self.progress_bar.value = (
                self.label_manager.current_index() / self.label_manager.image_count()
            )
