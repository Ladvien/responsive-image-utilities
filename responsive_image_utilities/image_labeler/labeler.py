import threading
import time
import flet as ft
from pynput import keyboard  # <--- changed here
from pynput.keyboard import Key, KeyCode
from rich import print

from responsive_image_utilities.image_labeler import LabelerConfig
from responsive_image_utilities.image_labeler.views.labeler_control import (
    ImageLabelerControlView,
)
from responsive_image_utilities.image_labeler.label_manager import LabelManager
from responsive_image_utilities.image_labeler.color_scheme import LabelerColorScheme


from flet import NavigationRailDestination as NavDest

from responsive_image_utilities.image_labeler.views.review_control import (
    ReviewControlView,
)


class LabelAppFactory:

    @staticmethod
    def create_labeler_app(config: LabelerConfig):
        def labeler_app(page: ft.Page):
            page.title = config.title
            page.window_width = config.window_width
            page.window_height = config.window_height
            page.window_resizable = config.window_resizable
            page.theme_mode = ft.ThemeMode.SYSTEM
            page.window.always_on_top = True
            page.window.focused = True
            page.window.full_screen = True

            color_scheme = LabelerColorScheme.flet_color_scheme()
            page.bgcolor = LabelerColorScheme.BACKGROUND
            page.theme = ft.Theme(color_scheme=color_scheme)
            page.theme_mode = ft.ThemeMode.DARK

            silent_focus = ft.TextField(
                autofocus=True,
                opacity=0.0,
                show_cursor=False,
                width=0,
                height=0,
            )

            label_manager = LabelManager(config.label_manager_config)
            if label_manager.unlabeled_count() == 0:
                page.add(ft.Text("No images found."))
                return

            image_labeler = ImageLabelerControlView(label_manager, color_scheme)
            labeled_image_pairs = label_manager.get_labeled_image_pairs()
            review = ReviewControlView(labeled_image_pairs, color_scheme)

            # Placeholder page content dict
            views = {
                0: review,
                1: image_labeler,
                # 2: ft.Text("About view (placeholder)", size=20),
            }

            content_area = ft.Container(content=views[0], expand=True)

            def switch_page(e: ft.ControlEvent):
                selected_index = e.control.selected_index
                content_area.content = views[selected_index]
                page.update()

            nav_rail = ft.NavigationRail(
                selected_index=0,
                label_type=ft.NavigationRailLabelType.ALL,
                extended=False,
                min_width=80,
                min_extended_width=200,
                destinations=[
                    NavDest(icon=ft.Icons.RATE_REVIEW, label="Review"),
                    NavDest(icon=ft.Icons.IMAGE, label="Labeling"),
                    # NavDest(icon=ft.icons.INFO, label="About"),
                ],
                on_change=switch_page,
            )

            layout = ft.Row(
                controls=[
                    nav_rail,
                    content_area,
                ],
                expand=True,
            )

            page.add(layout, silent_focus)
            page.update()
            silent_focus.focus()

            # Key handler integration
            def on_keyboard(key: Key | KeyCode):
                silent_focus.value = ""
                silent_focus.focus()

                if isinstance(content_area.content, ImageLabelerControlView):
                    handled = content_area.content.handle_keyboard_event(key)
                    if handled:
                        page.update()

                if isinstance(content_area.content, ReviewControlView):
                    handled = content_area.content.handle_keyboard_event(key)
                    if handled:
                        page.update()

            page.on_keyboard_event = on_keyboard

            pressed = False
            key_pressed: Key | KeyCode | None = None

            def on_press(key: Key | KeyCode):
                nonlocal pressed, key_pressed
                if page.window.focused:
                    pressed = True
                    key_pressed = key

            def on_release():
                nonlocal pressed, key_pressed
                if page.window.focused:
                    pressed = False
                    key_pressed = None

            def key_capture_loop():
                repeat_delay = config.key_press_debounce_delay
                while True:
                    if pressed:
                        on_keyboard(key_pressed)
                    time.sleep(repeat_delay)

            keyboard.Listener(on_press=on_press, on_release=on_release).start()
            threading.Thread(target=key_capture_loop, daemon=True).start()

        return labeler_app
