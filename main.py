from rich import print
from responsive_image_utilities import ImageLoader

loader = ImageLoader(
    "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
)

image = loader.load_images()[0]
print(image.test_resize_quality(50))
