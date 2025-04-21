import os
import csv
import flet as ft
from dataclasses import dataclass, asdict


@dataclass
class LabelerImageLoaderConfig:
    images_dir: str
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


class LabelerImageLoader:
    def __init__(self, config: LabelerImageLoaderConfig):
        self.config = config

    def total_count(self):
        return len(self.get_all_images())

    def get_next_image(self, current_index):
        if current_index >= self.total_count():
            return None
        return os.path.join(
            self.config.images_dir, self.get_all_images()[current_index]
        )

    def get_all_images(self):
        return sorted(
            [
                f
                for f in os.listdir(self.config.images_dir)
                if os.path.splitext(f)[1].lower() in self.config.allowed_exts
            ]
        )


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

            image_loader = LabelerImageLoader(config.image_loader_config)
            label_loader = LabelLoader(config.csv_path)

            if image_loader.total_count() == 0:
                page.add(ft.Text("No images found."))
                return

            # Load existing labels
            labeled = label_loader.get_labels()

            current_index = 0
            for i, img in enumerate(image_loader.get_all_images()):
                if img not in labeled:
                    current_index = i
                    break
            else:
                current_index = image_loader.total_count()

            # UI Elements
            img_display = ft.Image(width=700, height=500, fit=ft.ImageFit.CONTAIN)
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
                    img_display.src = None
                    progress_text.value = "✅ All images labeled!"
                    page.update()
                    return

                img_display.src = image_loader.get_next_image(idx)
                update_progress()
                page.update()

            def save_label(label):
                nonlocal current_index
                if current_index >= image_loader.total_count():
                    return
                img = image_loader.get_all_images()[current_index]
                labeled[img] = label
                with open(config.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([img, label])
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
                        img_display,
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
