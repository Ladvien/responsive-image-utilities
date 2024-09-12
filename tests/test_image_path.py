import pytest

from responsive_image_utilities.image_loader import ImageLoader

NUMBER_IMAGES_IN_FOLDER = 16
UPPERCASE_IMAGES_IN_FOLDER = 2


def test_image_loader_raises_exception_when_no_images_in_folder():
    with pytest.raises(Exception) as e:
        ImageLoader("tests/no_images", "tests/no_images")
    assert str(e.value) == "No files found in 'tests/no_images'."


def test_get_valid_image_paths_returns_only_images():
    loader = ImageLoader(
        "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
    )
    valid_images = loader.load_images()
    assert len(valid_images) == NUMBER_IMAGES_IN_FOLDER


def test_image_loader_load_images():
    folder = ImageLoader(
        "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
    )
    images = folder.load_images()

    assert len(images) == NUMBER_IMAGES_IN_FOLDER


def test_image_loader_loads_images_with_uppercase_extension():
    folder = ImageLoader(
        "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
    )
    images = folder.load_images()

    images_with_uppercase_ext = [
        image for image in images if image.potential_image.path.endswith("PNG")
    ]

    assert len(images_with_uppercase_ext) == UPPERCASE_IMAGES_IN_FOLDER
