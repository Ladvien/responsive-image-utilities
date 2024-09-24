from rich import print
from PIL import Image as PILImage
from responsive_image_utilities.image_loader import ImageLoader
from responsive_image_utilities.image_noiser import ImageNoiser

import numpy as np

loader = ImageLoader(
    "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
)

image_file = loader.load_images()[5]
print(np.array(image_file.image).shape)
result = image_file.test_resize_quality(300)
print(result)

resized_image = image_file.resize_by_width(400).image
resized_image.show()
# distorted_image = ImageNoiser.with_noise(image_file)
# distorted_image.show()
