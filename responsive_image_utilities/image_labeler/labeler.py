import threading
import time
import flet as ft
from pynput import keyboard  # <--- changed here
from pynput.keyboard import Key, KeyCode

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
            page.title = config.title
            page.window_width = config.window_width
            page.window_height = config.window_height
            page.window_resizable = config.window_resizable
            page.theme_mode = ft.ThemeMode.SYSTEM
            page.window.always_on_top = True
            page.window.focused = True

            silent_focus = ft.TextField(
                # disabled=True,
                # read_only=True,
                autofocus=True,
                opacity=0.0,
                show_cursor=False,
            )

            label_manager = LabelManager(config.label_manager_config)

            if label_manager.unlabeled_count() == 0:
                page.add(ft.Text("No images found."))
                return

            image_labeler = ImageLabelerControl(label_manager)

            def on_keyboard(key: Key | KeyCode):
                silent_focus.value = ""
                silent_focus.focus()
                handled = image_labeler.handle_keyboard_event(key)
                if handled:
                    page.update()

            # NOTE: This page handler has holes.  Like no
            # event for holding the key press.
            page.on_keyboard_event = on_keyboard

            page.add(
                image_labeler,
                silent_focus,
            )

            page.update()
            silent_focus.focus()

            #############################
            # Key Handling
            #############################

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

            # --- Background thread to update slider
            def key_capture_loop():
                repeat_delay = config.key_press_debounce_delay
                while True:
                    moved = False
                    if pressed:
                        on_keyboard(key_pressed)

                    if moved:
                        page.update()

                    time.sleep(repeat_delay)

            # Start key listener
            listener = keyboard.Listener(
                on_press=on_press,
                on_release=on_release,
            )
            listener.start()

            # Key listener
            threading.Thread(target=key_capture_loop, daemon=True).start()

            #############################
            # END: Key Handling
            #############################

        return labeler_app
