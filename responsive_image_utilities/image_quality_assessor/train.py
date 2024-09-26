from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from rich import print
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as torch_func
import numpy as np


class MLP(pl.LightningModule):
    def __init__(self, input_size, x_col="embeddings", y_col="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.x_col = x_col
        self.y_col = y_col
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.x_col]
        y = batch[self.y_col].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = torch_func.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.x_col]
        y = batch[self.y_col].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = torch_func.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@dataclass
class DataLoaderConfig:
    batch_size: int = 256
    num_workers: int = 0  # TODO: Dude, why can't I set this higher?
    shuffle: bool = True
    prediction_max_score = 10.0


@dataclass
class IQATrainerConfig:
    training_data_path: str
    test_data_path: str
    model_save_folder: str

    epochs: Optional[int] = 50
    model_input_size: Optional[int] = 1024
    model_save_name: Optional[str] = "linear_predictor_mse.pth"
    val_percentage: Optional[float] = 0.05

    model_save_path = None
    data_loader_config = DataLoaderConfig()

    def __post_init__(self):
        path = Path(self.model_save_folder + "/")
        path.mkdir(exist_ok=True, parents=True)

        self.model_save_path = path / self.model_save_name


class ImageQualityAssessmentTrainer:

    def __init__(
        self,
        config: IQATrainerConfig,
        data_loader_config: DataLoaderConfig = DataLoaderConfig(),
    ):
        # load the training data
        x = np.load(config.training_data_path)
        # Resize x to 768

        # x = x[:, : config.model_input_size]
        y = np.load(config.test_data_path)

        # split the data into train and validation
        train_border = int(x.shape[0] * (1 - config.val_percentage))

        # Transform to torch tensor
        train_tensor_x = torch.Tensor(x[:train_border])
        train_tensor_y = torch.Tensor(y[:train_border])

        # Create the data loader
        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_loader_config.batch_size,
            shuffle=data_loader_config.shuffle,
            num_workers=data_loader_config.num_workers,
        )

        val_tensor_x = torch.Tensor(x[train_border:])
        val_tensor_y = torch.Tensor(y[train_border:])

        # Create the validation data loader
        val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_loader_config.batch_size,
            num_workers=data_loader_config.num_workers,
        )

        # Define the model
        device = torch.device("mps")
        model = MLP(config.model_input_size).to(
            device
        )  # CLIP embedding dim is 768 for CLIP ViT L 14
        optimizer = torch.optim.Adam(model.parameters())

        # Choose the loss you want to optimize for
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()

        model.train()
        best_loss = 999

        for epoch in range(config.epochs):
            mse_losses = []
            mae_losses = []
            for batch_num, input_data in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = input_data
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                mse_losses.append(loss.item())

                optimizer.step()

                if batch_num % 1000 == 0:
                    print(
                        "\tEpoch %d | Batch %d | Loss %6.2f"
                        % (epoch, batch_num, loss.item())
                    )
                    # print(y)

            print("Epoch %d | Loss %6.2f" % (epoch, sum(mse_losses) / len(mse_losses)))
            mse_losses = []
            mae_losses = []

            for batch_num, input_data in enumerate(val_loader):
                optimizer.zero_grad()
                x, y = input_data
                x = x.to(device).float()
                y = y.to(device)

                output = model(x)
                loss = criterion(output, y)
                lossMAE = criterion2(output, y)
                # loss.backward()
                mse_losses.append(loss.item())
                mae_losses.append(lossMAE.item())
                # optimizer.step()

                if batch_num % 1000 == 0:
                    print(
                        "\tValidation - Epoch %d | Batch %d | MSE Loss %6.2f"
                        % (epoch, batch_num, loss.item())
                    )
                    print(
                        "\tValidation - Epoch %d | Batch %d | MAE Loss %6.2f"
                        % (epoch, batch_num, lossMAE.item())
                    )

                    # print(y)

            print(
                "Validation - Epoch %d | MSE Loss %6.2f"
                % (epoch, sum(mse_losses) / len(mse_losses))
            )
            print(
                "Validation - Epoch %d | MAE Loss %6.2f"
                % (epoch, sum(mae_losses) / len(mae_losses))
            )
            if sum(mse_losses) / len(mse_losses) < best_loss:
                print("Best MAE Val loss so far. Saving model")
                best_loss = sum(mse_losses) / len(mse_losses)
                print(best_loss)

                torch.save(model.state_dict(), config.model_save_path)

        torch.save(model.state_dict(), config.model_save_path)
        print(best_loss)

        print("Training done")
        print("Inference test with dummy samples from the val set, sanity check")

        model.eval()
        output = model(x[:5].to(device))

        print(output.size())
        print(output)


if __name__ == "__main__":

    config = IQATrainerConfig(
        training_data_path="training_data/ava_x_RN50x64.npy",
        test_data_path="training_data/ava_y_RN50x64.npy",
        model_save_folder="models/",
        model_save_name="linear_predictor_rn50x64_mse.pth",
    )

    ImageQualityAssessmentTrainer(config)
