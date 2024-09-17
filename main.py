from rich import print

from responsive_image_utilities.image_loader import ImageLoader
from responsive_image_utilities.image_noiser import ImageNoiser

loader = ImageLoader(
    "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
)

image_file = loader.load_images()[0]
# print(image.test_resize_quality(50))
distorted_image = ImageNoiser.add_noise(image_file, 1.0)
distorted_image.show()
