import flet as ft

from responsive_image_utilities.image_labeler import LabelerConfig
from responsive_image_utilities.image_labeler.controls.labeler_control import (
    ImageLabelerControl,
)
from responsive_image_utilities.image_labeler.label_manager import LabelManager


class LabelAppFactory:

    @staticmethod
    def create_labeler_app(config: LabelerConfig):
        """
        Create a labeler app using the provided configuration.
        """

        def labeler_app(page: ft.Page):
            # Setup
            page.title = config.title
            page.window_width = config.window_width
            page.window_height = config.window_height
            page.window_resizable = config.window_resizable
            page.theme_mode = ft.ThemeMode.SYSTEM
            page.window.always_on_top = True
            page.window.focused = True

            silent_focus = ft.TextField(
                visible=False,
                disabled=False,
                autofocus=True,
            )

            label_manager = LabelManager(config.label_manager_config)

            if label_manager.image_count() == 0:
                page.add(ft.Text("No images found."))
                return

            def on_key(event: ft.KeyboardEvent):
                image_labeler.handle_keyboard_event(event)

            image_labeler = ImageLabelerControl(label_manager)

            # Top level keyboard event handler
            def on_keyboard(e: ft.KeyboardEvent):
                image_labeler.handle_keyboard_event(e)

            page.on_keyboard_event = on_keyboard

            page.add(
                image_labeler,
                silent_focus,
            )

        return labeler_app
