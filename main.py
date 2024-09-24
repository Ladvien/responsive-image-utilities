from rich import print
from PIL import Image as PILImage
from responsive_image_utilities.image_loader import ImageLoader
from responsive_image_utilities.image_noiser import ImageNoiser

loader = ImageLoader(
    "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
)

image_file = loader.load_images()[5]
# print(image.test_resize_quality(50))
distorted_image = ImageNoiser.with_noise(image_file)
distorted_image.show()
