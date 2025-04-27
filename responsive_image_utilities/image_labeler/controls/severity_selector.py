from random import uniform
from typing import Callable
import flet as ft


class MinMaxSelector(ft.Row):
    def __init__(
        self,
        label: str,
        default_value: float,
        min_value: float = 0.0,
        max_value: float = 1.0,
        on_value_change: Callable | None = None,
    ):
        super().__init__()
        self.value = default_value
        self.label = ft.Text(
            label,
            size=10,
        )
        self.value_display = ft.Text(
            f"{self.value:.3f}",
            size=12,
            color=ft.colors.BLUE,
        )

        self.on_value_change = on_value_change

        self.controls = [
            ft.Column(
                [
                    self.label,
                    ft.Row(
                        [
                            self.value_display,
                            self.max_slider,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        tight=True,
                    ),
                ],
                tight=True,
                alignment=ft.MainAxisAlignment.START,
                spacing=0,
            )
        ]

    def _max_slider_change(self, e):
        self.value_display.value = f"{e.control.value:.3f}"
        self.max_slider.value = e.control.value
        self.value = e.control.value
        if self.on_value_change:
            self.on_value_change(self.value)

        self.update()
