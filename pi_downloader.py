import pandas as pd
import numpy as np
import requests
import os
import shutil
from multiprocessing.pool import ThreadPool
from random import shuffle, sample
from rich import print
"""
Should fire up a downloader in the background:
screen -Sdm  python pi_downloader.py 
"""

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
    index, hash, url = data
    try:
        if os.path.exists(os.path.join(OUTPUT_FOLDER, f"{hash}.jpg")):
            print(f"Skipping {url}, already downloaded.")
            return

        response = requests.get(url, stream=True, verify=False)
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



INPUT_FILE = "/media/ladvien/T7/laion-aesthetics-12m-umap-urls-and-hashes.csv"
OUTPUT_FOLDER = "/media/ladvien/T7/images/laion-aesthetics-12m-umap-images/"
SAMPLE_SIZE = 5000
NUMBER_OF_THREADS = 100
NUM_OF_ROWS = sum(1 for line in open(INPUT_FILE)) - 1
skip = sorted(sample(range(1, NUM_OF_ROWS + 1), NUM_OF_ROWS - SAMPLE_SIZE))
df = pd.read_csv(INPUT_FILE, skiprows=skip)

results = ThreadPool(8).imap_unordered(
    fetch_image, df.itertuples()
)
del df
[path for path in results]
