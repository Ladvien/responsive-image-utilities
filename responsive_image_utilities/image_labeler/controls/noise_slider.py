from typing import Tuple
import flet as ft


class PersistentLabeledRangeSlider(ft.Column):
    def __init__(
        self,
        initial_range: Tuple[float, float] = (0.0, 0.5),
        min_val: float = 0.0,
        max_val: float = 1.0,
        step: float = 0.001,
    ):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val
        self.step = step

        # Labels for the current start and end values
        self.start_value_label = ft.Text(
            f"{initial_range[0]:.2f}", text_align=ft.TextAlign.CENTER
        )
        self.end_value_label = ft.Text(
            f"{initial_range[1]:.2f}", text_align=ft.TextAlign.CENTER
        )

        self.range_slider = ft.RangeSlider(
            min=self.min_val,
            max=self.max_val,
            start_value=self._clamp(initial_range[0]),
            end_value=self._clamp(initial_range[1]),
            divisions=(
                int((self.max_val - self.min_val) / self.step) if self.step else None
            ),
            label="{value}",
            round=3,
            on_change=self._on_slider_change,
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
        self.start_value_label.value = f"{self.range_slider.start_value:.2f}"
        self.end_value_label.value = f"{self.range_slider.end_value:.2f}"
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
