import pytest

import responsive_image_utilities.image_loader as image_loader

NUMBER_VALID_IMAGES_IN_FOLDER = 16
UPPERCASE_IMAGES_IN_FOLDER = 2


def test_image_loader_raises_exception_when_no_images_in_folder():
    with pytest.raises(Exception) as e:
        image_loader.ImageLoader("tests/no_images", "tests/no_images")
    assert str(e.value) == "No files found in 'tests/no_images'."


def test_get_valid_image_paths_returns_only_images():
    loader = image_loader.ImageLoader(
        "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
    )
    valid_images = loader.load_images()
    assert len(valid_images) == NUMBER_VALID_IMAGES_IN_FOLDER


def test_image_loader_load_images():
    loader = image_loader.ImageLoader(
        "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
    )
    images = loader.load_images()

    assert len(images) == NUMBER_VALID_IMAGES_IN_FOLDER


def test_image_loader_loads_images_with_uppercase_extension():
    loader = image_loader.ImageLoader(
        "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
    )
    images = loader.load_images()

    images_with_uppercase_ext = [
        image for image in images if image.image_path.path.endswith("PNG")
    ]

    assert len(images_with_uppercase_ext) == UPPERCASE_IMAGES_IN_FOLDER


def test_image_loader_loads_all_valid_images():
    loader = image_loader.ImageLoader(
        "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
    )
    images = loader.load_images()

    assert all([isinstance(image, image_loader.ImageFile) for image in images])
    assert len(images) == NUMBER_VALID_IMAGES_IN_FOLDER
