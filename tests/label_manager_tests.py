import pytest
import csv
import os
import shutil
from pathlib import Path
from PIL import Image

from responsive_image_utilities.image_labeler.label_manager import (
    LabelManager,
    LabelWriter,
    LabeledImagePair,
    UnlabeledImagePair,
)
from responsive_image_utilities.image_labeler.label_manager_config import (
    LabelManagerConfig,
)
from responsive_image_utilities.image_utils.image_path import ImagePath

# Static Paths
STATIC_IMAGES_PATH = "tests/test_assets/images"
GENERATED_IMAGES_PATH = "tests/test_assets/generated"
TEMPORARY_IMAGES_OUTPUT_PATH = "tests/test_assets/output/temp"
TRAIN_IMAGES_OUTPUT_PATH = "tests/test_assets/output"
LABELS_CSV_PATH = TRAIN_IMAGES_OUTPUT_PATH + "/labels.csv"


@pytest.fixture
def setup_test_environment():
    """Clean output folders and create dummy images."""
    for path in [
        TRAIN_IMAGES_OUTPUT_PATH,
        TEMPORARY_IMAGES_OUTPUT_PATH,
        GENERATED_IMAGES_PATH,
    ]:
        if Path(path).exists():
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    dummy_image_paths = []
    for name in ["test_image.jpg", "test_image_noisy.jpg"]:
        img_path = os.path.join(GENERATED_IMAGES_PATH, name)
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(img_path)
        dummy_image_paths.append(img_path)

    yield dummy_image_paths

    for path in [
        TRAIN_IMAGES_OUTPUT_PATH,
        TEMPORARY_IMAGES_OUTPUT_PATH,
        GENERATED_IMAGES_PATH,
    ]:
        if Path(path).exists():
            shutil.rmtree(path)


@pytest.fixture
def labeler_manager_config(
    setup_test_environment,
):  # <- depends on setup_test_environment!
    return LabelManagerConfig(
        images_dir=GENERATED_IMAGES_PATH,
        output_dir=TRAIN_IMAGES_OUTPUT_PATH,
        temporary_dir=TEMPORARY_IMAGES_OUTPUT_PATH,
        label_csv_path=LABELS_CSV_PATH,
        overwrite_label_csv=True,
        severity_range=(0.1, 0.5),
    )


# -------------------- Tests ---------------------


def test_init_label_manager_returns_label_manager(
    labeler_manager_config, setup_test_environment
):
    label_manager = LabelManager(labeler_manager_config)
    assert isinstance(label_manager, LabelManager)


def test_label_writer_initializes_file(setup_test_environment):
    writer = LabelWriter(LABELS_CSV_PATH)
    assert os.path.exists(LABELS_CSV_PATH)

    with open(LABELS_CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["original_path", "noisy_path", "label"]


def test_label_writer_records_and_detects_label(setup_test_environment):
    label_writer = LabelWriter(LABELS_CSV_PATH)
    original = ImagePath(os.path.join(GENERATED_IMAGES_PATH, "test_image.jpg"))
    noisy = ImagePath(os.path.join(GENERATED_IMAGES_PATH, "test_image_noisy.jpg"))
    label = "good"

    labeled_pair = LabeledImagePair(original, noisy, label)
    label_writer.record_label(labeled_pair)

    assert label_writer.is_labeled(str(original))


def test_unlabeled_image_pair_label_returns_labeled_pair(setup_test_environment):
    original = ImagePath(os.path.join(GENERATED_IMAGES_PATH, "test_image.jpg"))
    noisy = ImagePath(os.path.join(GENERATED_IMAGES_PATH, "test_image_noisy.jpg"))
    unlabeled = UnlabeledImagePair(original, noisy)
    label = "awesome"

    labeled = unlabeled.label(label)
    assert isinstance(labeled, LabeledImagePair)
    assert labeled.label == "awesome"


def test_unlabeled_image_pair_label_raises_if_not_string(setup_test_environment):
    original = ImagePath(os.path.join(GENERATED_IMAGES_PATH, "test_image.jpg"))
    noisy = ImagePath(os.path.join(GENERATED_IMAGES_PATH, "test_image_noisy.jpg"))
    unlabeled = UnlabeledImagePair(original, noisy)

    with pytest.raises(Exception) as excinfo:
        unlabeled.label(999)  # invalid

    assert "Label must be a string" in str(excinfo.value)


def test_label_manager_save_label_adds_label(
    labeler_manager_config, setup_test_environment
):
    label_manager = LabelManager(labeler_manager_config)
    original = ImagePath(os.path.join(GENERATED_IMAGES_PATH, "test_image.jpg"))
    noisy = ImagePath(os.path.join(GENERATED_IMAGES_PATH, "test_image_noisy.jpg"))
    labeled_pair = LabeledImagePair(original, noisy, "ok")

    label_manager.save_label(labeled_pair)

    assert original in label_manager.labeled_image_paths

    with open(LABELS_CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert any(row["original_path"] == str(original) for row in rows)


def test_label_manager_save_label_raises_on_duplicate(
    labeler_manager_config, setup_test_environment
):
    label_manager = LabelManager(labeler_manager_config)
    original = ImagePath(os.path.join(GENERATED_IMAGES_PATH, "test_image.jpg"))
    noisy = ImagePath(os.path.join(GENERATED_IMAGES_PATH, "test_image_noisy.jpg"))
    labeled_pair = LabeledImagePair(original, noisy, "duplicate")

    label_manager.save_label(labeled_pair)

    with pytest.raises(Exception) as excinfo:
        label_manager.save_label(labeled_pair)

    assert "already labeled" in str(excinfo.value)


def test_label_manager_get_unlabeled_returns_pair_or_none(
    labeler_manager_config, setup_test_environment
):
    label_manager = LabelManager(labeler_manager_config)

    result = label_manager.new_unlabeled()
    assert result is None or isinstance(result, UnlabeledImagePair)
