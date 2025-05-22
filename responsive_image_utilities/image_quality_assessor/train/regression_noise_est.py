import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import wandb
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.models import create_model
from torch.cuda.amp import autocast as cuda_autocast, GradScaler


@dataclass
class Config:
    csv_path: str
    model_dir: str = "models"
    batch_size: int = 2  # Reduced from 4 to 2
    lr: float = 3e-4
    weight_decay: float = 5e-4
    epochs: int = 200
    val_frac: float = 0.1
    pretrained: bool = True
    device: str = None

    def __post_init__(self):
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"→ Using device: {self.device}")


def variable_size_collate(batch):
    filtered = [(img, label) for img, label in batch if img is not None]
    if not filtered:
        return [], torch.empty(0)
    images, labels = zip(*filtered)
    min_len = min(len(images), len(labels))
    return list(images[:min_len]), torch.stack(labels[:min_len])


class NoiseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_cols: List[str], transform=None):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((256, 256)),  # Reduce image size to save VRAM
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["noisy_image_path"]).convert("RGB")
        except FileNotFoundError:
            print(f"Image not found: {row['noisy_image_path']}")
            return None, None
        img = self.transform(img)
        labels = torch.tensor(
            [row[c] if pd.notna(row[c]) else 0.0 for c in self.label_cols],
            dtype=torch.float32,
        )
        return img, labels


def extract_patches(image: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    C, H, W = image.shape
    if H < patch_size or W < patch_size:
        raise ValueError(
            f"Image too small for patch extraction: {H}x{W} < {patch_size}"
        )

    unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(stride, stride))
    patches = unfold(image.unsqueeze(0))
    patches = patches.squeeze(0).transpose(0, 1)
    patches = patches.view(-1, C, patch_size, patch_size)
    return patches


class SwinTransformerRegressor(nn.Module):
    def __init__(self, num_outputs: int, fine_tune_backbone: bool = False):
        super().__init__()
        self.backbone = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0,
        )
        self.backbone.set_grad_checkpointing()

        embed_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
        )

        if not fine_tune_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.head.parameters():
                p.requires_grad = True

    def forward(self, batch_images: List[torch.Tensor]) -> torch.Tensor:
        batch_preds = []
        valid_images = []
        for img in batch_images:
            if img is None:
                continue
            try:
                patches = extract_patches(
                    img, patch_size=64, stride=64
                )  # Match model input resolution
                patches = nn.functional.interpolate(
                    patches, size=(224, 224), mode="bilinear", align_corners=False
                )
                if patches.shape[0] > 8:
                    patches = patches[:8]  # Limit number of patches
                if patches.shape[0] == 0:
                    continue
            except ValueError as e:
                print(e)
                continue
            with torch.amp.autocast(device_type="cuda"):
                feats = self.backbone(patches.to(img.device))
                pred = self.head(feats).mean(dim=0)
            batch_preds.append(pred)
            valid_images.append(img)
        if not batch_preds:
            return torch.empty(
                0, self.head[-1].out_features, device=batch_images[0].device
            )
        min_len = min(len(batch_preds), len(valid_images))
        return torch.stack(batch_preds[:min_len])


# Rest of the code remains unchanged...
def build_model(num_outputs: int, fine_tune_backbone: bool = False) -> nn.Module:
    return SwinTransformerRegressor(
        num_outputs=num_outputs,
        fine_tune_backbone=fine_tune_backbone,
    )


class MetricTracker:
    def __init__(self, label_cols: List[str]):
        self.label_cols = label_cols

    def compute(
        self, targets: np.ndarray, preds: np.ndarray, tag: str = ""
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        results = {}
        for i, col in enumerate(self.label_cols):
            y_true = targets[:, i]
            y_pred = preds[:, i]
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            try:
                auc = roc_auc_score((y_true > 0).astype(int), y_pred)
            except ValueError:
                auc = float("nan")
            results[col] = {"RMSE": rmse, "R2": r2, "AUC": auc}
            print(
                f"{tag.upper()} [{col:<24}] RMSE: {rmse:.4f} | R^2: {r2:.4f} | AUC: {auc:.4f}"
            )

        try:
            bin_targets = (targets > 0).astype(int)
            total_auc = roc_auc_score(bin_targets, preds, average="macro")
            print(f"{tag.upper()} [Total AUC]                 AUC: {total_auc:.4f}")
            results["total"] = {"AUC": total_auc}
        except ValueError:
            print(f"{tag.upper()} [Total AUC]                 AUC: N/A")
            results["total"] = {"AUC": float("nan")}

        return results, mean_squared_error(targets, preds)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.df = pd.read_csv(cfg.csv_path)
        self.df["split"] = self.df["split"].str.lower()
        self.label_cols = [
            c
            for c in self.df.columns
            if c not in ("original_image_path", "noisy_image_path", "split")
        ]

        wandb.init(
            project="noise-prediction", name="noise-prediction", config=vars(cfg)
        )

        train_df = self.df[self.df.split == "train"]
        test_df = self.df[self.df.split == "test"]
        val_df = train_df.sample(frac=cfg.val_frac, random_state=42)
        train_df = train_df.drop(val_df.index)

        self.train_loader = self._make_loader(train_df)
        self.val_loader = self._make_loader(val_df)
        self.test_loader = self._make_loader(test_df)

        self.model = build_model(
            num_outputs=len(self.label_cols), fine_tune_backbone=True
        ).to(cfg.device)

        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.lr,
            steps_per_epoch=len(self.train_loader),
            epochs=cfg.epochs,
            pct_start=0.1,
        )
        self.metric_tracker = MetricTracker(self.label_cols)

    def _make_loader(self, df: pd.DataFrame) -> DataLoader:
        ds = NoiseDataset(df, self.label_cols)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=variable_size_collate,
        )

    def train(self):
        best_val_loss = float("inf")
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            running_loss = 0.0
            for imgs, targets in tqdm(
                self.train_loader, desc=f"Epoch {epoch}/{self.cfg.epochs}"
            ):
                imgs = [img.to(self.cfg.device) for img in imgs]
                targets = targets.to(self.cfg.device)

                preds = self.model(imgs)
                if preds.size(0) == 0:
                    continue  # Skip empty batches

                min_len = min(preds.size(0), targets.size(0))
                loss = self.criterion(preds[:min_len], targets[:min_len])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()

            val_loss = self.evaluate(self.val_loader, tag="val")
            train_loss = running_loss / len(self.train_loader)
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(), Path(self.cfg.model_dir) / "best.pth"
                )
                print("  → New best model saved")

    def evaluate(self, loader: DataLoader, tag: str = "test") -> float:
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for imgs, y in loader:
                imgs = imgs.to(self.cfg.device)
                y_hat = self.model(imgs).cpu().numpy()
                preds.append(y_hat)
                targets.append(y.numpy())

        preds = np.vstack(preds)
        targets = np.vstack(targets)
        results, mse = self.metric_tracker.compute(targets, preds, tag=tag)

        metrics = {}
        for noise_type, m in results.items():
            for metric_name, value in m.items():
                metrics[f"{tag}/{noise_type}/{metric_name}"] = value
        metrics[f"{tag}/mse"] = mse
        wandb.log(metrics)

        return mse

    def run(self):
        self.train()
        print("\nFinal Test Evaluation:")
        self.evaluate(self.test_loader, tag="test")


if __name__ == "__main__":
    cfg = Config(csv_path="/mnt/datadrive_m2/ml_training_data/aiqa/noisy_labels.csv")
    trainer = Trainer(cfg)
    trainer.run()
