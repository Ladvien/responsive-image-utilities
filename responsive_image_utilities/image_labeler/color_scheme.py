# labeler_colors.py

import flet as ft


class LabelerColorScheme:
    """
    Defines a centralized color scheme for the Labeler App.
    """

    # Primary app colors
    PRIMARY = "#7F00FF"  # Vivid purple (buttons, highlights)
    SECONDARY = "#C792EA"  # Soft lavender (accents)
    BACKGROUND = "#1A002B"  # Very dark purple background
    SURFACE = "#24003A"  # Lighter card surface
    ERROR = "#FF5370"  # (Optional) Error color (soft red)

    # Text colors
    ON_PRIMARY = "#FFFFFF"  # Text on primary (white)
    ON_SECONDARY = "#FFFFFF"  # Text on secondary
    ON_BACKGROUND = "#FFFFFF"  # Default page text
    ON_SURFACE = "#CCCCCC"  # Card text

    @classmethod
    def flet_color_scheme(cls) -> ft.ColorScheme:
        """
        Return a Flet ColorScheme object based on these colors.
        """
        return ft.ColorScheme(
            primary=cls.PRIMARY,
            secondary=cls.SECONDARY,
            background=cls.BACKGROUND,
            surface=cls.SURFACE,
            error=cls.ERROR,
            on_primary=cls.ON_PRIMARY,
            on_secondary=cls.ON_SECONDARY,
            on_background=cls.ON_BACKGROUND,
            on_surface=cls.ON_SURFACE,
        )
