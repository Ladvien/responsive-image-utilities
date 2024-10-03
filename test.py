import pandas as pd

# df = pd.read_parquet("hf://datasets/dclure/laion-aesthetics-12m-umap/train.parquet")
# df.to_parquet("laion-aesthetics-12m-umap.parquet")

# df = pd.read_parquet("laion-aesthetics-12m-umap.parquet")


# for row in df.iterrows():
#     print(row)

import requests
import os
import shutil
from multiprocessing.pool import ThreadPool
from rich import print

from random import shuffle

OUTPUT_FOLDER = "/Users/ladvien/responsive-image-utilities/training_data/laion-aesthetics-12m-umap-images"


# df = pd.read_parquet("hf://datasets/dclure/laion-aesthetics-12m-umap/train.parquet")
# df.to_parquet("laion-aesthetics-12m-umap.parquet")

df = pd.read_parquet(
    "/Users/ladvien/responsive-image-utilities/training_data/laion-aesthetics-12m-umap.parquet"
)


def get_image_file_extension(url):
    if ".jpg" in url.lower():
        return ".jpg"
    elif ".jpeg" in url.lower():
        return ".jpeg"
    elif ".png" in url.lower():
        return ".png"
    elif ".gif" in url.lower():
        return ".gif"
    elif ".webp" in url.lower():
        return ".webp"
    else:
        return ".unknown"


url_and_hash = df.apply(lambda row: (row[0], row[8]), axis=1).tolist()
number_of_images = len(url_and_hash)
shuffle(url_and_hash)
del df

url_and_hash_of_files_not_yet_downloaded = [
    (url, hash)
    for url, hash in url_and_hash
    if not os.path.exists(os.path.join(OUTPUT_FOLDER, f"{hash}.jpg"))
]

number_of_images_left_to_download = len(url_and_hash_of_files_not_yet_downloaded)
print(
    f"The percentage of images left to download is: {number_of_images_left_to_download / number_of_images * 100:.2f}%"
)
input()


def fetch_image(data):
    url, hash = data
    try:

        if os.path.exists(os.path.join(OUTPUT_FOLDER, f"{hash}.jpg")):
            print(f"Skipping {url}, already downloaded.")
            return

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(
                os.path.join(OUTPUT_FOLDER, f"{hash}{get_image_file_extension(url)}"),
                "wb",
            ) as out_file:
                shutil.copyfileobj(response.raw, out_file)
                print(f"[green]Downloaded {url}[/green]")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        with open("failed.txt", "a") as f:
            f.write(f"{url}\n")


results = ThreadPool(8).imap_unordered(
    fetch_image, url_and_hash_of_files_not_yet_downloaded
)

[path for path in results]
