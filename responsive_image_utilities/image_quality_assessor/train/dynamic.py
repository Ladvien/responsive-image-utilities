from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score

class SiameseIQADataset(Dataset):
    def __init__(self, csv_file, transform=None, root_dir=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir
        self.label_map = {"unacceptable": 0, "acceptable": 1}

    def __len__(self):
        return len(self.df)
    
    def __safe_load(self, path) -> Image.Image:
        img = Image.open(path)

        # Handle palette images or RGBA images
        if img.mode == "P" or img.mode == "RGBA":
            img = img.convert("RGBA")
            # Discard alpha channel
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        return img


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        orig_path = Path(self.root_dir) / row["original_path"]
        noisy_path = Path(self.root_dir) / row["noisy_path"]


        orig_img = self.__safe_load(orig_path)
        noisy_img = self.__safe_load(noisy_path)

        if self.transform:
            orig_img = self.transform(orig_img)
            noisy_img = self.transform(noisy_img)

        label = torch.tensor([self.label_map[row["label"]]], dtype=torch.float32)
        return {"original": orig_img, "noisy": noisy_img, "label": label}


class SiameseResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, img1, img2):
        f1 = self.feature_extractor(img1).squeeze(-1).squeeze(-1)
        f2 = self.feature_extractor(img2).squeeze(-1).squeeze(-1)
        diff = torch.abs(f1 - f2)
        return self.classifier(diff)


@dataclass
class IQAConfig:
    csv_path: str
    model_save_folder: str
    root_dir: str = ""
    model_save_name: str = "siamese_resnet_binary_iqa.pth"
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 10
    val_split: float = 0.1
    test_split: float = 0.1
    learning_rate: float = 1e-3
    img_size: int = 512
    early_stopping_patience: int = 15
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

        transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        full_dataset = SiameseIQADataset(config.csv_path, transform, config.root_dir)

        total_size = len(full_dataset)
        test_size = int(total_size * config.test_split)
        val_size = int(total_size * config.val_split)
        train_size = total_size - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        self.model = SiameseResNetClassifier().to(config.device)
        self.optimizer = self.optimizer = optim.Adam([
            {"params": self.model.feature_extractor.parameters(), "lr": config.learning_rate * 0.1},
            {"params": self.model.classifier.parameters(), "lr": config.learning_rate},
        ])

        self.criterion = nn.BCEWithLogitsLoss()

        self.best_test_loss = float("inf")
        self.epochs_since_improvement = 0

    def train(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            train_losses = []

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}"):
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


            print(f"[Epoch {epoch + 1}]")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
            print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")


            if test_loss < self.best_test_loss:
                print(f"✅ New best test loss! Saving model to {self.config.model_save_path}")
                self.best_test_loss = test_loss
                torch.save(self.model.state_dict(), self.config.model_save_path)
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1
                print(f"⏳ No improvement. {self.epochs_since_improvement} epochs without improvement.")

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

        # Stack all batches together
        all_labels = torch.cat(all_labels).numpy()
        all_logits = torch.cat(all_logits).numpy()

        # Compute metrics
        preds = (torch.sigmoid(torch.tensor(all_logits)) > 0.5).float().numpy()
        acc = (preds == all_labels).mean()

        try:
            probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
            auc = roc_auc_score(all_labels, probs)
        except ValueError:
            # Happens if only one class in batch
            auc = float('nan')

        return avg_loss, acc, auc



if __name__ == "__main__":
    config = IQAConfig(
        csv_path="training_data/aiqa/labels.csv",
        model_save_folder="models",
        model_save_name="siamese_resnet_binary_iqa.pth",
        batch_size=32,
        epochs=500,
        early_stopping_patience=50,
        test_split=0.2,
    )

    trainer = ImageQualityClassifierTrainer(config)
    trainer.train()
