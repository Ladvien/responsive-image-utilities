# import flet as ft

# from responsive_image_utilities.image_labeler import LabelerAppConfig, LabelerApp

# config = LabelerAppConfig(
#     title="Image Labeler",
#     theme_mode=ft.ThemeMode.DARK,
#     padding=50,
#     icon_path="/icons/icon-512.png",
#     image_url="https://picsum.photos/200/200",
# )

# app_builder = LabelerApp(config)
# app = app_builder.build()

# ft.app(app)


import os, csv
import flet as ft

# **Configuration:** Set the directory containing images and the path for the CSV file.
IMAGES_DIR = "/Users/ladvien/ladvien.com/content/images"  # <- change this to your images folder path
CSV_PATH = os.path.join(IMAGES_DIR, "labels.csv")


def main(page: ft.Page):
    # Setup
    page.title = "Binary Image Labeler"
    page.window_width = 800
    page.window_height = 600
    page.window_resizable = True

    # Load images
    allowed_exts = {".jpg", ".jpeg", ".png"}
    all_images = sorted(
        [
            f
            for f in os.listdir(IMAGES_DIR)
            if os.path.splitext(f)[1].lower() in allowed_exts
        ]
    )
    total_count = len(all_images)
    if total_count == 0:
        page.add(ft.Text("No images found."))
        return

    # Load existing labels
    labeled = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    labeled[row[0]] = row[1]

    current_index = 0
    for i, img in enumerate(all_images):
        if img not in labeled:
            current_index = i
            break
    else:
        current_index = total_count

    # UI Elements
    img_display = ft.Image(width=700, height=500, fit=ft.ImageFit.CONTAIN)
    progress_text = ft.Text()
    progress_bar = ft.ProgressBar(width=300)

    def update_progress():
        progress_text.value = f"{len(labeled)}/{total_count} labeled"
        progress_bar.value = len(labeled) / total_count if total_count > 0 else 0.0

    def show_image(idx):
        if idx >= total_count:
            img_display.src = None
            progress_text.value = "✅ All images labeled!"
            page.update()
            return
        img_display.src = all_images[idx]
        update_progress()
        page.update()

    def save_label(label):
        nonlocal current_index
        if current_index >= total_count:
            return
        img = all_images[current_index]
        labeled[img] = label
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([img, label])
        current_index += 1
        show_image(current_index)

    def on_key(event: ft.KeyboardEvent):
        if event.key.upper() == "A":
            save_label("acceptable")
        elif event.key.upper() == "U":
            save_label("unacceptable")

        page.set_focus(silent_focus)

    page.on_keyboard_event = on_key

    # Hidden input field to suppress macOS "beep"
    # Note, if this field is made invisible, you will get
    # an annoying "beep" sound
    silent_focus = ft.TextField(visible=True, autofocus=True, width=0, height=0)

    # Info box for key commands
    shortcut_info = ft.Text(
        "Keyboard Shortcuts:\nA = Acceptable   |   U = Unacceptable",
        size=14,
        color=ft.colors.GREY_600,
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
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )

    show_image(current_index)


if __name__ == "__main__":
    ft.app(target=main, assets_dir=IMAGES_DIR)
