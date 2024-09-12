import pytest

from responsive_image_utilities.image_path import ImageFolder

NUMBER_IMAGES_IN_FOLDER = 16


def test_image_folder_raises_exception_when_no_images_in_folder():
    with pytest.raises(Exception) as e:
        ImageFolder("tests/no_images", "tests/no_images")
    assert str(e.value) == "No image files found in 'tests/no_images'."


def test_image_folder_load_images():
    folder = ImageFolder(
        "tests/test_assets/images/data-warehouse", "tests/test_assets/output"
    )
    images = folder.load_images()
    print(len(images))
    assert len(images) == NUMBER_IMAGES_IN_FOLDER
