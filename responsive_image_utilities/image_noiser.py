from dataclasses import dataclass
from PIL import Image as PILImage, ImageStat
import numpy as np
from dataclasses import dataclass, asdict
import torch
import piq
from PIL import Image as PILImage
import torchvision.transforms.functional as tf

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

    # TODO: Make these part of a configuration object
    # and loadable from file.
    min_gaussian_magnitude = 0
    max_gaussian_magnitude = 0.3

    max_color_channel_magnitude = 0.3

    max_spatially_correlated_magnitude = 0

    @classmethod
    def add_noise(
        cls, image_file: "ImageFile", magnitude: float = 1.0
    ) -> PILImage.Image:
        # image_file = cls.add_gaussian_noise(image_file.as_numpy_array(), magnitude)
        image = cls.add_spatially_correlated_noise(
            image_file.as_numpy_array(), magnitude
        )
        # image = cls.add_spatially_correlated_noise(image, magnitude)

        return image

    @classmethod
    def add_gaussian_noise(
        cls, image_array: np.ndarray, magnitude: float = 1.0
    ) -> PILImage.Image:
        """
        Add Gaussian noise to an image, including the color channels.
        """
        mean = ImageStat.Stat(PILImage.fromarray(image_array)).mean[0]
        std_dev = ImageStat.Stat(PILImage.fromarray(image_array)).stddev[0]
        magnitude = map_value(magnitude, 0, 1, 0, cls.max_gaussian_magnitude)
        std_dev = std_dev * magnitude
        noise = np.random.normal(mean, std_dev, image_array.shape)
        noisy_image = image_array + noise

        return PILImage.fromarray(noisy_image.astype(np.uint8))

    @classmethod
    def add_spatially_correlated_noise(
        cls, image_array: np.ndarray, magnitude: float = 1.0
    ) -> PILImage.Image:
        # Compute filter kernel with radius correlation_scale (can probably be a bit smaller)
        correlation_scale = 50
        x = np.arange(-correlation_scale, correlation_scale)
        y = np.arange(-correlation_scale, correlation_scale)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X * X + Y * Y)
        filter_kernel = np.exp(-(dist**2) / (2 * correlation_scale))

        # Generate n-by-n grid of spatially correlated noise
        n = 50
        noise = np.random.randn(n, n)

        # Perform the convolution using FFT
        noise_fft = np.fft.fft2(noise)
        filter_kernel_fft = np.fft.fft2(filter_kernel, s=noise.shape)
        convolved = np.fft.ifft2(noise_fft * filter_kernel_fft)

        # Take the real part of the convolved result
        noise = np.real(convolved)

        # Convert to PIL image
        noise = np.clip(noise, -1, 1)
        noise = (noise + 1) / 2 * 255
        noise = noise.astype(np.uint8)

        return PILImage.fromarray(noise)
