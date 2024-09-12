from PIL import Image as PILImage

from responsive_image_utilities.image_loader import ImageFileData


class Image:

    def __imit__(self, data: ImageFileData):
        self.image = data.image
        self.data = data.potential_image
