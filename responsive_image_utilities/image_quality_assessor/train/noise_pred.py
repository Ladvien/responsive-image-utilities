from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# Dataset
# =========================


class NoiseParameterDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file_or_df):
        if isinstance(csv_file_or_df, (str, Path)):
            self.df = pd.read_csv(csv_file_or_df)
        else:
            self.df = csv_file_or_df.reset_index(drop=True)

        self.param_cols = [
            col
            for col in self.df.columns
            if col not in ["original_image_path", "noisy_image_path", "label", "split"]
        ]

        self.label_map = {"unacceptable": 0, "acceptable": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        thresholds = [
            row[col] if pd.notna(row[col]) else 0.0 for col in self.param_cols
        ]
        x = torch.tensor(thresholds, dtype=torch.float32)
        y_class = torch.tensor([self.label_map[row["label"]]], dtype=torch.float32)
        return {"x": x, "label": y_class}


# =========================
# Model
# =========================


class NoiseClassifierRegressor(nn.Module):
    def __init__(self, num_params):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(num_params, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.class_output = nn.Linear(32, 1)
        self.reg_output = nn.Linear(32, num_params)

    def forward(self, x):
        features = self.hidden(x)
        return self.class_output(features), self.reg_output(features)


# =========================
# Config
# =========================


@dataclass
class IQAConfig:
    csv_path: str
    model_save_folder: str
    model_save_name: str = "noise_multitask.pth"
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 50
    val_split: float = 0.1
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    device: str = None

    def __post_init__(self):
        Path(self.model_save_folder).mkdir(parents=True, exist_ok=True)
        self.model_save_path = Path(self.model_save_folder) / self.model_save_name
        if torch.backends.mps.is_available():
            print("✅ Using Apple Silicon GPU")
            self.device = "mps"
        elif torch.cuda.is_available():
            print("✅ Using CUDA GPU")
            self.device = "cuda"
        else:
            print("✅ Using CPU")
            self.device = "cpu"


# =========================
# Trainer
# =========================


class NoiseTrainer:
    def __init__(self, config: IQAConfig):
        self.config = config
        df = pd.read_csv(config.csv_path)
        df["split"] = df["split"].str.lower()

        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()

        val_df = train_df.sample(frac=config.val_split, random_state=42)
        train_df = train_df.drop(val_df.index).reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        self.train_ds = NoiseParameterDataset(train_df)
        self.val_ds = NoiseParameterDataset(val_df)
        self.test_ds = NoiseParameterDataset(test_df)

        self.param_cols = self.train_ds.param_cols
        print(f"✅ Found {len(self.param_cols)} noise parameters: {self.param_cols}")

        self.train_loader = DataLoader(
            self.train_ds, batch_size=config.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(self.val_ds, batch_size=config.batch_size)
        self.test_loader = DataLoader(self.test_ds, batch_size=config.batch_size)

        self.model = NoiseClassifierRegressor(len(self.param_cols)).to(config.device)
        self.class_loss_fn = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        self.best_val_loss = float("inf")
        self.epochs_since_improvement = 0

    def train(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}"
            ):
                x = batch["x"].to(self.config.device)
                y_class = batch["label"].to(self.config.device)

                self.optimizer.zero_grad()
                class_logits, reg_preds = self.model(x)
                class_loss = self.class_loss_fn(class_logits, y_class)
                reg_loss = self.reg_loss_fn(reg_preds, x)
                loss = class_loss + 0.5 * reg_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            val_loss, val_acc, val_auc = self.evaluate(self.val_loader)
            test_loss, test_acc, test_auc = self.evaluate(self.test_loader)

            print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
            print(
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
            )
            print(
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}"
            )

            if val_loss < self.best_val_loss:
                print(f"✅ New best model — saving to {self.config.model_save_path}")
                self.best_val_loss = val_loss
                self.epochs_since_improvement = 0
                torch.save(self.model.state_dict(), self.config.model_save_path)
            else:
                self.epochs_since_improvement += 1
                print(f"⏳ No improvement. Patience: {self.epochs_since_improvement}")

            if self.epochs_since_improvement >= self.config.early_stopping_patience:
                print("🛑 Early stopping triggered.")
                break

    def evaluate(self, loader):
        self.model.eval()
        losses, all_labels, all_preds = [], [], []

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.config.device)
                y_class = batch["label"].to(self.config.device)
                class_logits, reg_preds = self.model(x)
                class_loss = self.class_loss_fn(class_logits, y_class)
                reg_loss = self.reg_loss_fn(reg_preds, x)
                losses.append((class_loss + 0.5 * reg_loss).item())
                probs = torch.sigmoid(class_logits).cpu().numpy()
                all_preds.extend(probs.flatten())
                all_labels.extend(y_class.cpu().numpy().flatten())

        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        acc = (preds_binary == np.array(all_labels)).mean()

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = float("nan")

        return np.mean(losses), acc, auc


# =========================
# Main
# =========================

if __name__ == "__main__":
    config = IQAConfig(
        csv_path="/Users/ladvien/responsive_images_workspace/responsive-image-utilities/responsive_image_utilities/training_data/aiqa/noisy_labels.csv",
        model_save_folder="models",
        epochs=500,
        early_stopping_patience=10,
    )

    trainer = NoiseTrainer(config)
    trainer.train()

    trainer.model.load_state_dict(torch.load(config.model_save_path))
    trainer.model.eval()

    batch = next(iter(trainer.test_loader))
    x = batch["x"].to(config.device)
    y_class = batch["label"].to(config.device)

    with torch.no_grad():
        class_logits, reg_preds = trainer.model(x)
        class_probs = torch.sigmoid(class_logits).cpu().numpy()
        reg_preds_np = reg_preds.cpu().numpy()

    true_labels = y_class.cpu().numpy().flatten()

    print("\n==== SAMPLE PREDICTIONS ====")
    for i in range(min(10, len(batch["x"]))):
        print(f"Sample {i+1}")
        print(
            f"  True label:        {'acceptable' if true_labels[i] == 1 else 'unacceptable'}"
        )
        print(f"  Predicted prob:    {class_probs[i][0]:.4f}")
        print(f"  True thresholds:   {batch['x'][i].cpu().numpy()}")
        print(f"  Predicted thresh.: {reg_preds_np[i]}")
        print("")

    fig, axes = plt.subplots(
        1, len(trainer.param_cols), figsize=(5 * len(trainer.param_cols), 4)
    )
    if len(trainer.param_cols) == 1:
        axes = [axes]

    for idx, col in enumerate(trainer.param_cols):
        ax = axes[idx]
        true_vals = batch["x"][:, idx].cpu().numpy()
        pred_vals = reg_preds_np[:, idx]
        ax.scatter(true_vals, pred_vals)
        ax.set_title(col)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.plot([0, 1], [0, 1], "r--")

    plt.tight_layout()
    plt.show()

    print("\n🖼️ Visualizing predictions on test set...")
    sample_df = trainer.test_ds.df.sample(n=6, random_state=42).reset_index(drop=True)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    for i, ax in enumerate(axs.flatten()):
        row = sample_df.iloc[i]
        img_path = Path(row["noisy_image_path"])
        label = row["label"]

        param_tensor = (
            torch.tensor(
                [row[col] if pd.notna(row[col]) else 0.0 for col in trainer.param_cols],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .to(config.device)
        )

        with torch.no_grad():
            class_logit, _ = trainer.model(param_tensor)
            prob = torch.sigmoid(class_logit).item()
            pred_label = "acceptable" if prob > 0.5 else "unacceptable"

        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(
                f"{img_path.name}\nTrue: {label} | Pred: {pred_label} ({prob:.2f})",
                fontsize=10,
                color="green" if label == pred_label else "red",
            )
        except FileNotFoundError:
            ax.axis("on")
            ax.set_title(f"Missing:\n{img_path.name}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()
