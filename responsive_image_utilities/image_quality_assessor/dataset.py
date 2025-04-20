import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class ContinuousIQADataset(Dataset):
    def __init__(self, csv_file, transform=None, root_dir=None):
        """
        A PyTorch dataset for continuous image quality assessment (IQA) tasks.

        This dataset reads image paths and their corresponding quality scores from a CSV file.
                image_path,score
                blog_images/photo1.jpg,1.00
                blog_images/photo2.jpg,0.82
                blog_images/photo3.jpg,0.10

        Args:
            csv_file (str): Path to CSV file with columns: image_path, score
            transform (callable, optional): Transform to apply to images
            root_dir (str, optional): Optional base path to prepend to image paths
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        score = torch.tensor([float(row["score"])], dtype=torch.float32)
        return image, score
