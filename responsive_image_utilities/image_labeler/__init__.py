import flet as ft
from dataclasses import dataclass


@dataclass
class LabelerAppConfig:
    title: str = "Labeler App"
    theme_mode: ft.ThemeMode = ft.ThemeMode.LIGHT
    padding: int = 50
    icon_path: str = "/icons/icon-512.png"
    image_url: str = "https://picsum.photos/200/200"


class LabelerApp:
    def __init__(self, config: LabelerAppConfig = LabelerAppConfig()):
        super().__init__()
        self.config = config

    def build(self):
        def labeler_app(page: ft.Page):
            page.title = self.config.title
            page.theme_mode = self.config.theme_mode
            page.padding = self.config.padding
            page.update()

            # img = ft.Image(
            #     src=f"/icons/icon-512.png",
            #     width=100,
            #     height=100,
            #     fit=ft.ImageFit.CONTAIN,
            # )
            # images = ft.Row(expand=1, wrap=False, scroll="always")

            # page.add(img, images)

            # for i in range(0, 30):
            #     images.controls.append(
            #         ft.Image(
            #             src=f"https://picsum.photos/200/200?{i}",
            #             width=200,
            #             height=200,
            #             fit=ft.ImageFit.NONE,
            #             repeat=ft.ImageRepeat.NO_REPEAT,
            #             border_radius=ft.border_radius.all(10),
            #         )
            #     )
            page.update()

        return labeler_app
