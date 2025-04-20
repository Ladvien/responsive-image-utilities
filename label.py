import flet as ft

from responsive_image_utilities.image_labeler import LabelerAppConfig, LabelerApp

config = LabelerAppConfig(
    title="Image Labeler",
    theme_mode=ft.ThemeMode.DARK,
    padding=50,
    icon_path="/icons/icon-512.png",
    image_url="https://picsum.photos/200/200",
)

app_builder = LabelerApp(config)
app = app_builder.build()

ft.app(app)
