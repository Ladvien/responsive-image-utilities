import os
import csv
import flet as ft
from dataclasses import dataclass, asdict
from random import uniform
from rich import print

from responsive_image_utilities.image_utils.image_loader import ImageLoader
from responsive_image_utilities.image_utils.image_noiser import ImageNoiser
from responsive_image_utilities.image_utils.image_path import ImagePath


@dataclass
class LabelerImageLoaderConfig:
    images_dir: str
    output_dir: str
    allowed_exts: list | None = None

    def __post_init__(self):
        if self.allowed_exts is None:
            self.allowed_exts = [".jpg", ".jpeg", ".png", ".gif"]
        else:
            self.allowed_exts = [ext.lower() for ext in self.allowed_exts]
        self.allowed_exts = list(set(self.allowed_exts))

        if not os.path.exists(self.images_dir):
            raise ValueError(f"Directory {self.images_dir} does not exist.")
        if not os.path.isdir(self.images_dir):
            raise ValueError(f"{self.images_dir} is not a directory.")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.output_dir):
            raise ValueError(f"{self.output_dir} is not a directory.")


class LabelLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def get_labels(self) -> dict:
        labels = {}
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        labels[row[0]] = row[1]

        return labels

    def save_label(self, img, label):
        self.labels[img] = label
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([img, label])


@dataclass
class BinaryLabelerPageConfig:
    title: str = "Binary Image Labeler"
    window_width: int = 800
    window_height: int = 700
    window_resizable: bool = True
    theme_mode: ft.ThemeMode = ft.ThemeMode.DARK

    # ImageLoaderConfig
    image_loader_config: LabelerImageLoaderConfig | None = None

    # Label path
    csv_path: str | None = None

    # Noise settings
    noise_functions: list = None
    severity_range: tuple = (0.20, 0.95)

    def __post_init__(self):
        if self.csv_path is None:
            self.csv_path = "labels.csv"

        if self.image_loader_config is None:
            self.image_loader_config = LabelerImageLoaderConfig()

        # Check if the images directory exists
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "label"])

        # Check if the CSV file is empty
        self.image_loader_config.images_dir = os.path.abspath(
            self.image_loader_config.images_dir
        )
        self.csv_path = os.path.abspath(self.csv_path)


class LabelAppFactory:

    @staticmethod
    def create_labeler_app(config: BinaryLabelerPageConfig):
        """
        Create a labeler app using the provided configuration.
        """

        def labeler_app(page: ft.Page):
            # Setup
            page.title = config.title
            page.window_width = config.window_width
            page.window_height = config.window_height
            page.window_resizable = config.window_resizable
            page.window.maximized = True

            image_loader = ImageLoader(
                config.image_loader_config.images_dir,
                config.image_loader_config.output_dir,
            )
            label_loader = LabelLoader(config.csv_path)

            if image_loader.total_count() == 0:
                page.add(ft.Text("No images found."))
                return

            # Load existing labels
            labeled = label_loader.get_labels()

            current_index = 0
            for i, img in enumerate(image_loader.get_all_image_paths()):
                if img not in labeled:
                    current_index = i
                    break
            else:
                current_index = image_loader.total_count()

            # UI Elements
            original_image_display = ft.Image(
                width=700, height=500, fit=ft.ImageFit.CONTAIN
            )
            noisy_image_display = ft.Image(
                width=700, height=500, fit=ft.ImageFit.CONTAIN
            )

            progress_text = ft.Text()
            progress_bar = ft.ProgressBar(width=300)

            def update_progress():
                progress_text.value = (
                    f"{len(labeled)}/{image_loader.total_count()} labeled"
                )
                progress_bar.value = (
                    len(labeled) / image_loader.total_count()
                    if image_loader.total_count() > 0
                    else 0.0
                )

            def show_image(idx):
                if idx >= image_loader.total_count():
                    noisy_image_display.src = None
                    progress_text.value = "✅ All images labeled!"
                    page.update()
                    return

                image_path = image_loader.get_image_path(idx)

                new_image = ImagePath(image_path.path).load()
                # TODO: May not need.
                # new_image_path = os.path.join(
                #     config.image_loader_config.output_dir, image_path.name
                # )
                noisy_image_path = os.path.join(
                    config.image_loader_config.output_dir,
                    f"{image_path.name}_noisy.jpg",
                )

                min_noise, max_noise = config.severity_range
                noise_level = uniform(min_noise, max_noise)
                noisy_image = ImageNoiser.add_jpeg_compression(new_image, noise_level)
                noisy_image.save(noisy_image_path, quality=95)

                original_image_display.src_base64 = image_path.load_as_base64()
                noisy_image_display.src_base64 = ImagePath(
                    noisy_image_path
                ).load_as_base64()

                update_progress()
                page.update()

            def save_label(label: str, image_path: str):
                nonlocal current_index
                if current_index >= image_loader.total_count():
                    return
                image_path = image_loader.get_image_path(current_index).path
                labeled[image_path] = label
                with open(config.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([image_path, label])
                current_index += 1
                show_image(current_index)

            def on_key(event: ft.KeyboardEvent):
                # Right arrow key to go to the next image
                if event.key == "Arrow Right":
                    save_label("acceptable")
                elif event.key == "Arrow Left":
                    save_label("unacceptable")

            page.on_keyboard_event = on_key

            # Hidden input field to suppress macOS "beep"
            # Note, if this field is made invisible, you will get
            # an annoying "beep" sound
            silent_focus = ft.TextField(visible=True, autofocus=True, width=0, height=0)

            # Info box for key commands
            shortcut_info = ft.Text(
                "Keyboard Shortcuts:\n <- Left Arrow: Unacceptable\n -> Right Arrow: Acceptable",
                size=14,
                color=ft.Colors.GREY_600,
                text_align=ft.TextAlign.CENTER,
            )

            page.add(
                ft.Column(
                    [
                        ft.Row(
                            [original_image_display, noisy_image_display],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                        shortcut_info,
                        ft.Row([progress_text], alignment=ft.MainAxisAlignment.CENTER),
                        progress_bar,
                        silent_focus,  # keep this invisible input at the bottom
                    ],
                    expand_loose=True,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                )
            )

            show_image(current_index)

        return labeler_app
