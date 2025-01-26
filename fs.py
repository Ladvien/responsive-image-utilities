
from shutil import copyfile
import os
from rich import print
from itertools import islice
from pathlib import Path


FILES_PER_DIR = 10000
INPUT_DIR = "/media/ladvien/T7/images/buffer"
OUTPUT_DIR = "/media/ladvien/T7/images/tmp"
RECORD_CSV = "/media/ladvien/T7/images/tmp/catalog.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(RECORD_CSV):
    with open(RECORD_CSV, "w") as f:
        f.write("path,filename\n")


copy_dir = ""

existing_paths = list(Path(OUTPUT_DIR).rglob("*"))
index = len(existing_paths)

new_paths = list(Path(INPUT_DIR).rglob("*"))
all_paths = new_paths  + existing_paths

print(f"Starting at index: {index}")
print(f"Number of existing files: {index}")
print(f"Number of existing plus buffer files: {len(all_paths)}")

with open(RECORD_CSV, "a") as f:
    while True:
        image_paths = all_paths[index:index + FILES_PER_DIR]
        for path in image_paths:

            if index % FILES_PER_DIR == 0 or not copy_dir:
                copy_dir = f"{OUTPUT_DIR}/laion_{index}"
                os.makedirs(copy_dir, exist_ok=True)
                print(f"Creating directory {copy_dir}")

            path = Path(path)
            filename = path.name

            destination_path = f"{copy_dir}/{filename}"

            index += 1
            if os.path.exists(destination_path):
                print(f"File {destination_path} already exists. Skipping.")
                continue

            print(f"Copying {path} to {destination_path}")
            copyfile(path, destination_path)

            relative_destination_path = destination_path.replace(OUTPUT_DIR, "")
            f.write(f"{relative_destination_path},{filename}\n")
