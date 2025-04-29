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
    test_split: float = 0.1
    learning_rate: float = 1e-4
    img_size: int = 224
    early_stopping_patience: int = 5
    device: str | None = None

    def __post_init__(self):
        Path(self.model_save_folder).mkdir(parents=True, exist_ok=True)
        self.model_save_path = Path(self.model_save_folder) / self.model_save_name
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"


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

        full_dataset = BinaryIQADataset(config.csv_path, transform, config.root_dir)

        total_size = len(full_dataset)
        test_size = int(total_size * config.test_split)
        val_size = int(total_size * config.val_split)
        train_size = total_size - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        self.model = CNNBinaryClassifier().to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_test_loss = float("inf")
        self.epochs_since_improvement = 0

    def train(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            train_losses = []

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
                train_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)
            val_loss, val_acc = self.evaluate(self.val_loader)
            test_loss, test_acc = self.evaluate(self.test_loader)

            print(f"[Epoch {epoch + 1}]")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

            if test_loss < self.best_test_loss:
                print(
                    f"✅ New best test loss! Saving model to {self.config.model_save_path}"
                )
                self.best_test_loss = test_loss
                torch.save(self.model.state_dict(), self.config.model_save_path)
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1
                print(
                    f"⏳ No improvement. {self.epochs_since_improvement} epochs without improvement."
                )

            if self.epochs_since_improvement >= self.config.early_stopping_patience:
                print("🛑 Early stopping triggered.")
                break

    def evaluate(self, loader):
        self.model.eval()
        losses = []
        correct, total = 0, 0

        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(self.config.device)
                y = batch["label"].to(self.config.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)
                losses.append(loss.item())

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_loss = sum(losses) / len(losses)
        acc = correct / total
        return avg_loss, acc
