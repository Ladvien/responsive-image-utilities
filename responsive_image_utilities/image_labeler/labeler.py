import flet as ft

from responsive_image_utilities.image_labeler import (
    LabelerConfig,
)
from responsive_image_utilities.image_labeler.controls.image_display import (
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
            # page.window.maximized = True

            # Hidden input field to suppress macOS "beep"
            # Note, if this field is made invisible, you will get
            # an annoying "beep" sound
            silent_focus = ft.TextField(visible=True, autofocus=True, width=0, height=0)

            label_manager = LabelManager(config.label_manager_config)

            if label_manager.image_count() == 0:
                page.add(ft.Text("No images found."))
                return

            image_labeler = ImageLabelerControl(
                label_manager,
            )

            def on_key(event: ft.KeyboardEvent):
                if label_manager.current_index() >= label_manager.image_count():
                    return

                if event.key == "Arrow Right":
                    label_manager.save_label("acceptable")
                elif event.key == "Arrow Left":
                    label_manager.save_label("unacceptable")

                image_labeler.update_content()
                image_labeler.update()

            page.on_keyboard_event = on_key

            page.add(
                image_labeler,
                silent_focus,
            )

        return labeler_app
