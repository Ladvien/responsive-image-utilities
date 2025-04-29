# labeler_colors.py

import flet as ft


class LabelerColorScheme:
    """
    Defines a centralized color scheme for the Labeler App.
    """

    # Core colors
    PRIMARY = "#D72638"  # Cherry red (primary action)
    SECONDARY = "#00F0FF"  # Electric cyan (accents and sliders)
    BACKGROUND = "#121212"  # Deep matte black (background)
    SURFACE = "#1E1E1E"  # Card / surface background
    ERROR = "#FF5370"  # Soft red (for errors, warnings)

    # Text and foreground
    ON_PRIMARY = "#FFFFFF"  # Text on cherry red
    ON_SECONDARY = "#000000"  # Text on electric cyan
    ON_BACKGROUND = "#E0E0E0"  # General body text
    ON_SURFACE = "#CCCCCC"  # Subdued card text

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
