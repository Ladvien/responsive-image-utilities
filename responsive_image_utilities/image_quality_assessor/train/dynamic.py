from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from timm import create_model
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BinaryIQADataset(Dataset):
    def __init__(self, csv_file, transform=None, root_dir=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir
        self.label_map = {"unacceptable": 0, "acceptable": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["noisy_path"]

        if self.root_dir:
            img_path = Path(self.root_dir) / img_path

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor([self.label_map[row["label"]]], dtype=torch.float32)
        return {"image": image, "label": label}


class CNNBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = create_model("mobilenetv2_100", pretrained=True, num_classes=1)
        base.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(base.classifier.in_features, 1)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


@dataclass
class IQAConfig:
    csv_path: str
    model_save_folder: str
    root_dir: str = ""
    model_save_name: str = "mobilenetv2_regressor.pth"
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 10
    val_split: float = 0.1
    learning_rate: float = 1e-4
    img_size: int = 224
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        Path(self.model_save_folder).mkdir(parents=True, exist_ok=True)
        self.model_save_path = Path(self.model_save_folder) / self.model_save_name


class ImageQualityClassifierTrainer:
    def __init__(self, config: IQAConfig):
        self.config = config

        transform = transforms.Compose(
            [
                transforms.Resize((config.img_size, config.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = BinaryIQADataset(config.csv_path, transform, config.root_dir)
        val_size = int(len(dataset) * config.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        self.model = CNNBinaryClassifier().to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self):
        best_loss = float("inf")

        for epoch in range(self.config.epochs):
            self.model.train()
            losses = []

            for batch in tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}"
            ):
                x = batch["image"].to(self.config.device)
                y = batch["label"].to(self.config.device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            avg_train_loss = sum(losses) / len(losses)
            print(f"[Epoch {epoch + 1}] Train BCE Loss: {avg_train_loss:.4f}")

            val_loss, val_acc = self.evaluate()
            print(
                f"[Epoch {epoch + 1}] Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}"
            )

            if val_loss < best_loss:
                print(f"✅ New best model! Saving to {self.config.model_save_path}")
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.config.model_save_path)

    def evaluate(self):
        self.model.eval()
        losses = []
        correct, total = 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["image"].to(self.config.device)
                y = batch["label"].to(self.config.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)
                losses.append(loss.item())

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)

        if len(losses) == 0:
            print("⚠️ Warning: Validation set is empty or malformed.")
            return None

        return sum(losses) / len(losses), correct / total
