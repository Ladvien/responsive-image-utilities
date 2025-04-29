from typing import Callable, Tuple
import flet as ft


class PersistentLabeledRangeSlider(ft.Column):
    def __init__(
        self,
        initial_range: Tuple[float, float] = (0.0, 0.5),
        min_val: float = 0.0,
        max_val: float = 1.0,
        step: float = 0.001,
        on_end_change: Callable = None,
    ):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.on_change_end = on_end_change

        # Labels for the current start and end values
        self.start_value_label = ft.Text(
            f"{initial_range[0] * 100}%", text_align=ft.TextAlign.CENTER
        )
        self.end_value_label = ft.Text(
            f"{initial_range[1] * 100}%", text_align=ft.TextAlign.CENTER
        )

        self.range_slider = ft.RangeSlider(
            min=self.min_val,
            max=self.max_val,
            start_value=self._clamp(initial_range[0]),
            end_value=self._clamp(initial_range[1]),
            # TODO: Disables knobs
            # divisions=(
            #     int((self.max_val - self.min_val) / self.step) if self.step else None
            # ),
            # label="{value}",
            round=3,
            on_change=self._on_slider_change,
            on_change_end=self._on_end_change,
            expand=True,
        )

        # Row to display labels aligned with the slider thumbs
        self.labels_row = ft.Row(
            controls=[
                self.start_value_label,
                ft.Container(expand=True),  # Spacer
                self.end_value_label,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        self.controls = [
            self.labels_row,
            self.range_slider,
        ]

        self.spacing = 5
        self.alignment = ft.MainAxisAlignment.CENTER

    def _clamp(self, value: float) -> float:
        """Clamp value between min and max."""
        return min(max(value, self.min_val), self.max_val)

    def _on_slider_change(self, e: ft.ControlEvent):
        """Update labels when slider values change."""
        self.start_value_label.value = (
            f"{round(self.range_slider.start_value * 100, 2)}%"
        )
        self.end_value_label.value = f"{round(self.range_slider.end_value * 100, 2)}%"
        self.labels_row.update()

    @property
    def value(self) -> Tuple[float, float]:
        """Current range (start_value, end_value)."""
        return (self.range_slider.start_value, self.range_slider.end_value)

    @value.setter
    def value(self, new_range: Tuple[float, float]):
        """Set a new (start, end) range."""
        lower, upper = new_range
        self.range_slider.start_value = self._clamp(lower)
        self.range_slider.end_value = self._clamp(upper)
        self.range_slider.update()
        self._on_slider_change(None)  # Update labels immediately

    def _on_end_change(self, e: ft.ControlEvent):
        """Handle end of slider change."""
        if self.range_slider.on_change_end:
            self.on_change_end(
                e, self.range_slider.start_value, self.range_slider.end_value
            )
