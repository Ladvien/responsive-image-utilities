from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import random
from torchvision.models import mobilenet_v3_small


# =========================
# Dataset
# =========================


class PairedTransform:
    def __init__(self, img_size=224, augment=True):
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(5),
                ]
            )
        else:
            self.transform = transforms.Resize((img_size, img_size))

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, img1, img2):
        seed = random.randint(0, 1000000)
        random.seed(seed)
        torch.manual_seed(seed)
        img1 = self.transform(img1)
        random.seed(seed)
        torch.manual_seed(seed)
        img2 = self.transform(img2)
        return self.to_tensor(img1), self.to_tensor(img2)


class SiameseIQADataset(Dataset):
    def __init__(self, csv_file_or_df, transform=None, root_dir=None):
        if isinstance(csv_file_or_df, (str, Path)):
            full_path = Path(csv_file_or_df)
            self.df = pd.read_csv(full_path.resolve())
        elif isinstance(csv_file_or_df, pd.DataFrame):
            self.df = csv_file_or_df.reset_index(drop=True)
        else:
            raise ValueError("csv_file_or_df must be a file path or a DataFrame.")

        self.transform = transform
        self.root_dir = root_dir
        self.label_map = {"unacceptable": 0, "acceptable": 1}

    def __len__(self):
        return len(self.df)

    def __safe_load(self, path) -> Image.Image:
        img = Image.open(path)
        if img.mode in ["P", "RGBA"]:
            img = img.convert("RGBA").convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        orig_path = Path(self.root_dir) / row["original_image_path"]
        noisy_path = Path(self.root_dir) / row["noisy_image_path"]

        orig_img = self.__safe_load(orig_path)
        noisy_img = self.__safe_load(noisy_path)

        if self.transform:
            orig_img, noisy_img = self.transform(orig_img, noisy_img)

        label = torch.tensor([self.label_map[row["label"]]], dtype=torch.float32)
        return {"original": orig_img, "noisy": noisy_img, "label": label}


# =========================
# Siamese ViT Model
# =========================


class SiameseViTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        base_cnn = mobilenet_v3_small(weights="DEFAULT")

        base_cnn.classifier = nn.Identity()

        self.feature_extractor = base_cnn

        embed_dim = 576  # mobilenet_v3_small outputs 576-dim features

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, img1, img2):
        f1 = self.feature_extractor(img1)
        f2 = self.feature_extractor(img2)
        combined = torch.cat([f1, f2, torch.abs(f1 - f2)], dim=-1)
        return self.classifier(combined)


# =========================
# Config
# =========================


@dataclass
class IQAConfig:
    csv_path: str
    model_save_folder: str
    root_dir: str = ""
    model_save_name: str = "siamese_vit_binary_iqa.pth"
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 20
    val_split: float = 0.1
    test_split: float = 0.1
    learning_rate: float = 1e-4
    img_size: int = 224  # ViT requires 224x224
    early_stopping_patience: int = 10
    device: str | None = None

    def __post_init__(self):
        Path(self.model_save_folder).mkdir(parents=True, exist_ok=True)
        self.model_save_path = Path(self.model_save_folder) / self.model_save_name
        if torch.backends.mps.is_available():
            print("Using Apple Silicon GPU")
            self.device = "mps"
        elif torch.cuda.is_available():
            print("Using GPU")
            self.device = "cuda"
        else:
            print("Using CPU")
            self.device = "cpu"


# =========================
# Trainer
# =========================


class ImageQualityClassifierTrainer:
    def __init__(self, config: IQAConfig):
        self.config = config

        # Paired transforms (only augment train)
        train_transform = PairedTransform(config.img_size, augment=True)
        eval_transform = PairedTransform(
            config.img_size, augment=False
        )  # disable augmentation

        df = pd.read_csv(config.csv_path)

        # ===============================
        # Stratified Splitting ✅
        # ===============================
        train_df, test_df = train_test_split(
            df, test_size=config.test_split, stratify=df["label"], random_state=42
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=config.val_split,
            stratify=train_df["label"],
            random_state=42,
        )

        # ===============================
        # Build datasets for each split ✅
        # ===============================
        self.train_dataset = SiameseIQADataset(
            train_df, train_transform, config.root_dir
        )
        self.val_dataset = SiameseIQADataset(val_df, eval_transform, config.root_dir)
        self.test_dataset = SiameseIQADataset(test_df, eval_transform, config.root_dir)

        # ===============================
        # Build loaders ✅
        # ===============================
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
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

        # ===============================
        # Class imbalance weight ✅
        # ===============================
        neg = (train_df["label"] == "unacceptable").sum()
        pos = (train_df["label"] == "acceptable").sum()

        pos_weight_value = max(neg / pos, 1.0) if pos > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(
            config.device
        )

        print(f"🔎 pos_weight = {pos_weight_value:.4f} (neg={neg}, pos={pos})")

        # ===============================
        # Model, Optimizer, Loss ✅
        # ===============================
        self.model = SiameseViTClassifier().to(config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.best_test_loss = float("inf")
        self.epochs_since_improvement = 0

    def train(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            train_losses = []

            for batch in tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}"
            ):
                x1 = batch["original"].to(self.config.device)
                x2 = batch["noisy"].to(self.config.device)
                y = batch["label"].to(self.config.device)

                self.optimizer.zero_grad()
                logits = self.model(x1, x2)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)
            val_loss, val_acc, val_auc = self.evaluate(self.val_loader)
            test_loss, test_acc, test_auc = self.evaluate(self.test_loader)

            print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}")
            print(
                f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
            )
            print(
                f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}"
            )

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
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in loader:
                x1 = batch["original"].to(self.config.device)
                x2 = batch["noisy"].to(self.config.device)
                y = batch["label"].to(self.config.device)

                logits = self.model(x1, x2)
                loss = self.criterion(logits, y)
                losses.append(loss.item())
                all_labels.append(y.cpu())
                all_logits.append(logits.cpu())

        avg_loss = sum(losses) / len(losses)
        all_labels = torch.cat(all_labels).numpy()
        all_logits = torch.cat(all_logits).numpy()

        preds = (torch.sigmoid(torch.tensor(all_logits)) > 0.5).float().numpy()
        acc = (preds == all_labels).mean()

        try:
            probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
            auc = roc_auc_score(all_labels, probs)
        except ValueError:
            auc = float("nan")

        return avg_loss, acc, auc


# =========================
# Run training
# =========================

if __name__ == "__main__":
    config = IQAConfig(
        csv_path="training_data/seeds.csv",
        model_save_folder="models",
        model_save_name="siamese_vit_binary_iqa.pth",
        epochs=50,
        early_stopping_patience=10,
        test_split=0.2,
    )

    trainer = ImageQualityClassifierTrainer(config)
    trainer.train()
