from __future__ import annotations  # Needed for type hinting

from PIL import Image as PILImage, ImageStat
import numpy as np

# from imagecorruptions import corrupt
from PIL import Image as PILImage
from responsive_image_utilities.utils import map_value


class ImageNoiser:
    """
    # 1. Additive Gaussian noise -> Done
    # 2. Additive noise in color channels -> Done
    # 3. Spatially correlated noise
    # 4. Masked noise
    # 5. High frequency noise
    # 6. Impulse noise
    # 7. Quantization noise
    # 8. Gaussian blur
    # 9. Image denoising
    # 10 JPEG compression
    # 11. JPEG2000 compression
    # 12. JPEG transmission errors
    # 13. JPEG2000 transmission errors
    # 14. Non eccentricity pattern noise
    # 15. Local block-wise distortions of different intensity
    # 16. Mean shift (intensity shift)
    # 17. Contrast change
    """

    corruptions = [
        # I just eyeballed these to see which ones looked good
        "gaussian_noise",
        # "gaussian_blur",
        "jpeg_compression",
        # "contrast",
        # "elastic_transform",
        # "shot_noise",
        # "impulse_noise",
        # "defocus_blur",
        # "glass_blur",
        # "motion_blur",
        # "zoom_blur",
        # "snow",
        # "frost",
        # "fog",
        # "brightness",
        # "pixelate",
        # "speckle_noise",
        # "spatter",
        # "saturate",
    ]

    @classmethod
    def with_noise(cls, image_file: ImageFile, severity: float = 0.2) -> PILImage.Image:  # type: ignore  # noqa: F821

        image = image_file.as_numpy_array()
        image_to_corrupt = image.copy()

        image_to_corrupt = cls.add_gaussian_noise(image_to_corrupt, severity)
        image_to_corrupt = cls.add_jpeg_compression(image_to_corrupt, severity)

        return image_to_corrupt

    @classmethod
    def add_gaussian_noise(
        cls, image: PILImage.Image, severity: float = 0.2
    ) -> PILImage.Image:
        """
        Add Gaussian noise to an image, including the color channels.
        """
        image_array = np.array(np.array(image)) / 255.0
        image_array = np.clip(
            image_array, np.random.normal(size=image_array.shape, scale=severity), 1.0
        )

        return PILImage.fromarray((image_array * 255).astype(np.uint8))

    @classmethod
    def add_jpeg_compression(
        cls, image: PILImage.Image, severity: float = 0.9
    ) -> PILImage.Image:
        quality = int(map_value(severity, 1, 0, 0, 95))
        image.save("temp.jpg", quality=quality)
        return PILImage.open("temp.jpg")

    # @classmethod
    # def add_spatially_correlated_noise(
    #     cls, image_array: np.ndarray, magnitude: float = 1.0
    # ) -> PILImage.Image:
    #     width, height, color_channels = image_array.shape

    #     print(width, height, color_channels)
    #     noise = generate_perlin_noise_2d((width, height), (4, 4))

    #     noise = noise * 255
    #     return PILImage.fromarray(noise)
    #     # return PILImage.fromarray(noisy_image.astype(np.uint8))
