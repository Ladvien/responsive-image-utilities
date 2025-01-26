<<<<<<< HEAD
<<<<<<< HEAD
import pandas as pd

# df = pd.read_parquet("hf://datasets/dclure/laion-aesthetics-12m-umap/train.parquet")
# df.to_parquet("laion-aesthetics-12m-umap.parquet")

# df = pd.read_parquet("laion-aesthetics-12m-umap.parquet")


# for row in df.iterrows():
#     print(row)

=======
=======
>>>>>>> ee54e15c268b68457a35ba701f03d33ccae4221f


import pandas as pd
import numpy as np
<<<<<<< HEAD
>>>>>>> ee54e15c268b68457a35ba701f03d33ccae4221f
=======
>>>>>>> ee54e15c268b68457a35ba701f03d33ccae4221f
import requests
import os
import shutil
from multiprocessing.pool import ThreadPool
<<<<<<< HEAD
<<<<<<< HEAD

from random import shuffle

# OUTPUT_FOLDER = "/srv/nfs/datadrive/images/laion-aesthetics-12m-umap-images"
OUTPUT_FOLDER = "/mnt/datadrive/images/laion-aesthetics-12m-umap-images"


# df = pd.read_parquet("hf://datasets/dclure/laion-aesthetics-12m-umap/train.parquet")
# df.to_parquet("laion-aesthetics-12m-umap.parquet")

df = pd.read_parquet(
    "/mnt/datadrive/laion-aesthetics-12m-umap-urls-and-hashes.csv"
)


# def get_image_file_extension(url):
#     if ".jpg" in url.lower():
#         return ".jpg"
#     elif ".jpeg" in url.lower():
#         return ".jpeg"
#     elif ".png" in url.lower():
#         return ".png"
#     elif ".gif" in url.lower():
#         return ".gif"
#     elif ".webp" in url.lower():
#         return ".webp"
#     else:
#         return ".unknown"


# url_and_hash = df.apply(lambda row: (row[0], row[8]), axis=1).tolist()
# number_of_images = len(url_and_hash)
# shuffle(url_and_hash)
# del df

# url_and_hash_of_files_not_yet_downloaded = [
#     (url, hash)
#     for url, hash in url_and_hash
#     if not os.path.exists(os.path.join(OUTPUT_FOLDER, f"{hash}.jpg"))
# ]

# number_of_images_left_to_download = len(url_and_hash_of_files_not_yet_downloaded)
# print(
#     f"The percentage of images left to download is: {number_of_images_left_to_download / number_of_images * 100:.2f}%"
# )
# input()


# def fetch_image(data):
#     url, hash = data
#     try:

#         if os.path.exists(os.path.join(OUTPUT_FOLDER, f"{hash}.jpg")):
#             print(f"Skipping {url}, already downloaded.")
#             return

#         response = requests.get(url, stream=True)
#         if response.status_code == 200:
#             with open(
#                 os.path.join(OUTPUT_FOLDER, f"{hash}{get_image_file_extension(url)}"),
#                 "wb",
#             ) as out_file:
#                 shutil.copyfileobj(response.raw, out_file)
#                 print(f"[green]Downloaded {url}[/green]")
#     except Exception as e:
#         print(f"Failed to download {url}: {e}")
#         with open("failed.txt", "a") as f:
#             f.write(f"{url}\n")


# results = ThreadPool(8).imap_unordered(
#     fetch_image, url_and_hash_of_files_not_yet_downloaded
# )

# [path for path in results]

print(df.head())
=======
=======
>>>>>>> ee54e15c268b68457a35ba701f03d33ccae4221f
from random import shuffle, sample
from rich import print

"""
Should fire up a downloader in the background:
screen -Sdm  python pi_downloader.py 
"""

INPUT_FILE = "/media/ladvien/T7/merged.csv"
OUTPUT_FOLDER = "/media/ladvien/T7/images/buffer/"
# DOWNLOADED_CATALOG = "/media/ladvien/T7/images/tmp/catalog.csv"
SAMPLE_SIZE = 50000
NUMBER_OF_THREADS = 100
NUM_OF_ROWS = sum(1 for line in open(INPUT_FILE)) - 1

# Sample 1000 rows of csv
# existing_file_list = pd.read_csv(DOWNLOADED_CATALOG)

# print(existing_file_list["filename"].tolist())


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


def fetch_image(data):
    index, hash, url, existing_filename = data
    try:
        image_extensions = ["jpg", "jpeg", "png", "gif"]
        for ext in image_extensions:
            if os.path.exists(os.path.join(OUTPUT_FOLDER, f"{hash}.{ext}")):
                print(f"Skipping {url}, already downloaded.")
                return

        response = requests.get(url, stream=True, verify=False, timeout=3)
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


skip = sorted(sample(range(1, NUM_OF_ROWS + 1), NUM_OF_ROWS - SAMPLE_SIZE))
df = pd.read_csv(INPUT_FILE, skiprows=skip, usecols=["hash", "URL", "filename"])
df = df[(df['filename'] == '') | (df['filename'].isna())]


results = ThreadPool(8).imap_unordered(fetch_image, df.itertuples())
del df
[path for path in results]
<<<<<<< HEAD
>>>>>>> ee54e15c268b68457a35ba701f03d33ccae4221f
=======
>>>>>>> ee54e15c268b68457a35ba701f03d33ccae4221f
