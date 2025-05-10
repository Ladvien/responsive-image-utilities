import os
import pandas as pd
from pathlib import Path
from uuid import uuid4
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split

from image_utils.image_noiser import ImageNoiser
from image_utils.utils import map_value

# ==========================================
# CONFIGURATION
# ==========================================

CSV_PATH = "/Users/ladvien/responsive_images_workspace/responsive-image-utilities/responsive_image_utilities/training_data/aiqa/seeds.csv"
OUTPUT_CSV = Path(CSV_PATH).parent / "noisy_labels.csv"
TRAIN_DIR = Path(CSV_PATH).parent / "train"
TEST_DIR = Path(CSV_PATH).parent / "test"
TRAIN_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.2  # 20% test set

print(f"✅ Output dirs:\n- Train: {TRAIN_DIR}\n- Test:  {TEST_DIR}")

# ------------------------------------------
# Load Data
# ------------------------------------------

df = pd.read_csv(CSV_PATH)

# Load previous output (if exists)
if OUTPUT_CSV.exists():
    df_existing = pd.read_csv(OUTPUT_CSV)
    processed_paths = set(df_existing["original_image_path"])
    records = df_existing.to_dict(orient="records")
    print(f"🕵️‍♂️ Skipping {len(processed_paths)} previously processed images.")
else:
    processed_paths = set()
    records = []

# ------------------------------------------
# Train/Test Split (based on original image path)
# ------------------------------------------

unique_orig_paths = df["original_image_path"].dropna().unique()
train_paths, test_paths = train_test_split(
    unique_orig_paths, test_size=TEST_SIZE, random_state=42
)
train_paths_set = set(train_paths)
test_paths_set = set(test_paths)

print(f"🔀 Split: {len(train_paths)} train, {len(test_paths)} test")

# ------------------------------------------
# Noise Functions
# ------------------------------------------

NOISE_FN_MAP = {
    "add_gaussian_noise": ImageNoiser.add_gaussian_noise,
    "add_jpeg_compression": ImageNoiser.add_jpeg_compression,
    "add_gaussian_blur": ImageNoiser.add_gaussian_blur,
}
ALL_NOISE_FUNCTIONS = list(NOISE_FN_MAP.keys())


def apply_noise_pipeline(
    image: PILImage.Image, operations: list[tuple[str, float]]
) -> PILImage.Image:
    img = image.convert("RGB") if image.mode != "RGB" else image.copy()
    for fn_name, severity in operations:
        fn = NOISE_FN_MAP.get(fn_name)
        if fn:
            img = fn(img, severity)
        else:
            print(f"⚠️  Unknown noise function '{fn_name}' — skipping.")
    return img


# ------------------------------------------
# Main Processing Loop
# ------------------------------------------

for _, row in df.iterrows():
    orig_path = Path(row["original_image_path"])
    if not orig_path.exists():
        print(f"❌ Missing image: {orig_path}")
        continue

    if str(orig_path) in processed_paths:
        continue

    label = row["label"]
    image = PILImage.open(orig_path)

    # Parse noise ops with order
    ordered_ops = []
    fn_severity_dict = {fn: 0.0 for fn in ALL_NOISE_FUNCTIONS}

    for i in range(1, 11):
        fn_name = row.get(f"fn_{i}_name")
        threshold = row.get(f"fn_{i}_threshold")
        order = row.get(f"fn_{i}_order")

        if pd.notna(fn_name) and pd.notna(threshold) and pd.notna(order):
            ordered_ops.append((int(order), fn_name, float(threshold)))
            fn_severity_dict[fn_name] = float(threshold)

    if not ordered_ops:
        print(f"⚠️  No valid noise operations for {orig_path}")
        continue

    # Apply noise
    ordered_ops.sort()
    pipeline_ops = [(fn, sev) for _, fn, sev in ordered_ops]
    noisy_image = apply_noise_pipeline(image, pipeline_ops)

    # Determine split group
    group = "train" if str(orig_path) in train_paths_set else "test"
    dest_dir = TRAIN_DIR if group == "train" else TEST_DIR

    # Save noisy image
    output_filename = f"{uuid4()}_{orig_path.stem}_noisy.jpg"
    output_path = dest_dir / output_filename

    if output_path.exists():
        print(f"⚠️  Already exists: {output_path}")
        continue

    noisy_image.save(output_path, quality=95)
    print(f"✅ Saved: {group}/{output_filename}")

    # Record output
    record = {
        "original_image_path": str(orig_path),
        "noisy_image_path": str(output_path),
        "label": label,
        "split": group,
        **fn_severity_dict,
    }
    records.append(record)

# ------------------------------------------
# Save Output CSV
# ------------------------------------------

pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
print(f"📝 Saved noisy image labels to: {OUTPUT_CSV}")
