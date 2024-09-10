from dataclasses import dataclass
from PIL import Image as PILImage


@dataclass
class ImageAttributes:
    width: int
    height: int
    mode: str
    format: str
    size: int
