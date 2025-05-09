import os
import pandas as pd
from pathlib import Path
from uuid import uuid4
from PIL import Image as PILImage

from image_utils.image_noiser import ImageNoiser
from image_utils.utils import map_value

# ==========================================
# CONFIGURATION
# ==========================================

CSV_PATH = "/Users/ladvien/responsive_images_workspace/adaptive_labeler/training_data/aiqa/seeds.csv"
OUTPUT_CSV = Path(CSV_PATH).parent / "noisy_labels.csv"
TRAINING_DIR = Path(CSV_PATH).parent / "training"
TRAINING_DIR.mkdir(exist_ok=True)

print(f"✅ Training images will be saved to: {TRAINING_DIR}")

# Load seed CSV
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
# Noise function mapping
# ------------------------------------------

NOISE_FN_MAP = {
    "add_gaussian_noise": ImageNoiser.add_gaussian_noise,
    "add_jpeg_compression": ImageNoiser.add_jpeg_compression,
    "add_gaussian_blur": ImageNoiser.add_gaussian_blur,
}

ALL_NOISE_FUNCTIONS = list(NOISE_FN_MAP.keys())

# ------------------------------------------
# Noise application
# ------------------------------------------


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


# ==========================================
# MAIN LOOP
# ==========================================

for _, row in df.iterrows():
    orig_path = Path(row["original_image_path"])
    if not orig_path.exists():
        print(f"❌ Missing image: {orig_path}")
        continue

    if str(orig_path) in processed_paths:
        continue  # Skip previously processed

    label = row["label"]
    image = PILImage.open(orig_path)

    # Parse noise ops
    operations = []
    fn_severity_dict = {fn: 0.0 for fn in ALL_NOISE_FUNCTIONS}

    for i in range(1, 11):
        fn_name = row.get(f"fn_{i}")
        threshold = row.get(f"fn_{i}_threshold")
        if pd.notna(fn_name) and pd.notna(threshold):
            operations.append((fn_name, float(threshold)))
            fn_severity_dict[fn_name] = float(threshold)

    if not operations:
        print(f"⚠️  No noise operations for {orig_path}")
        continue

    noisy_image = apply_noise_pipeline(image, operations)

    # Output filename
    output_filename = f"{uuid4()}_{orig_path.stem}_noisy.jpg"
    output_path = TRAINING_DIR / output_filename

    # Avoid overwriting (edge case, unlikely)
    if output_path.exists():
        print(f"⚠️  Skipping already existing: {output_path}")
        continue

    noisy_image.save(output_path, quality=95)
    print(f"✅ Saved: {output_path.name}")

    # Record entry
    record = {
        "original_image_path": str(orig_path),
        "noisy_image_path": str(output_path),
        "label": label,
        **fn_severity_dict,
    }
    records.append(record)

# ==========================================
# SAVE UPDATED CSV
# ==========================================

pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
print(f"📝 Saved noisy image labels to: {OUTPUT_CSV}")
