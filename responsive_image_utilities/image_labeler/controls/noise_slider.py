from typing import Callable
import flet as ft
from pynput.keyboard import Key, KeyCode


from flet import SliderInteraction


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
        self._on_change_end = on_change_end

        self.slider = ft.Slider(
            min=self.min_val,
            max=self.max_val,
            value=self._clamp(initial_value),
            divisions=None,
        )

        self.controls = [self.slider]
        self.alignment = ft.MainAxisAlignment.CENTER

    def _clamp(self, value: float) -> float:
        """Clamp value within min and max."""
        return min(max(value, self.min_val), self.max_val)

    @property
    def value(self) -> float:
        """Current value of the slider."""
        return float(self.slider.value)

    @value.setter
    def value(self, new_value: float):
        self.slider.value = self._clamp(new_value)
        self.slider.update()

    def handle_keyboard_event(self, key: Key | KeyCode):
        """Handle arrow key events. Returns True if handled."""

        if isinstance(key, KeyCode):
            return False

        if key.name == "up":
            self.value += self.step
            if self._on_change_end:
                self._on_change_end(self.value)
            return True

        if key.name == "down":
            self.value -= self.step
            if self._on_change_end:
                self._on_change_end(self.value)
            return True

        return False
