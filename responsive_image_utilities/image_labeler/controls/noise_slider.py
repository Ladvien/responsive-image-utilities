from typing import Callable
import flet as ft


class KeyboardBasedSlider(ft.Row):
    def __init__(
        self,
        initial_value: float = 0.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        step: float = 0.01,
        on_change_end: Callable[[float], None] = None,
    ):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self._on_change_end_callback = on_change_end

        self._slider = ft.Slider(
            min=min_val,
            max=max_val,
            value=initial_value,
            divisions=self._calculate_divisions(),
            # on_focus=self._on_focus,
            # on_blur=self._on_blur,
        )

        self.controls = [
            ft.Column(
                [
                    self._slider,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=1,
                tight=True,
                expand=True,
            )
        ]

        self._has_focus = False

    def build(self):
        return self

    def _calculate_divisions(self):
        total_range = self.max_val - self.min_val
        if (
            self.step
            and self.step > 0
            and abs((total_range / self.step) - round(total_range / self.step)) < 1e-9
        ):
            return int(round(total_range / self.step))
        return None

    @property
    def value(self) -> float:
        return float(self._slider.value or 0.0)

    @value.setter
    def value(self, new_value: float):
        val = float(new_value)
        val = min(max(val, self.min_val), self.max_val)
        self._slider.value = val
        try:
            self._slider.update()
        except Exception:
            pass

    def _increment_value(self):
        new_val = min((self._slider.value or 0) + (self.step or 0), self.max_val)
        if new_val != self._slider.value:
            self._slider.value = new_val
            self._slider.update()
            self._fire_on_change_end()

    def _decrement_value(self):
        new_val = max((self._slider.value or 0) - (self.step or 0), self.min_val)
        if new_val != self._slider.value:
            self._slider.value = new_val
            self._slider.update()
            self._fire_on_change_end()

    def _fire_on_change_end(self):
        if self._on_change_end_callback:
            self._on_change_end_callback(self.value)

    # def _on_focus(self, e: ft.OnFocusEvent):
    #     self._has_focus = True

    # def _on_blur(self, e: ft.OnFocusEvent):
    #     self._has_focus = False

    def handle_keyboard_event(self, e: ft.KeyboardEvent) -> bool:
        """Handles keyboard event. Returns True if handled."""
        # if not self._has_focus:
        #     return False

        if e.key == "Arrow Up":
            self._increment_value()
            return True
        elif e.key == "Arrow Down":
            self._decrement_value()
            return True

        return False
