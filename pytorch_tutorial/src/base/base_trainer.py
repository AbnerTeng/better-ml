from abc import abstractmethod
from typing import Optional, Dict, Any, Union

import numpy as np
from rich import print
from rich.progress import track
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils.tools import EarlyStopping


class BaseTrainer:
    """
    Base class for training neural networks

    Args:
        model: nn.Module
        settings: Dict, default=None -> settings for training
    """
    def __init__(
        self,
        model: nn.Module,
        settings: Optional[Dict[str, Any]] = None
    ) -> None:
        self.model = model
        self.settings = settings
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.loss = {
            "train": [],
            "val": [],
        }

    @abstractmethod
    def train_step(
        self,
        feature: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        *args,
        **kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, dataloader: DataLoader, *args, **kwargs):
        raise NotImplementedError

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> nn.Module:
        early_stopping = EarlyStopping(
            patience=self.settings.patience,
            verbose=self.settings.verbose,
        )
        for epoch in track(range(self.settings.n_epochs)):
            train_loss = []
            self.model.train()

            for feature, target in train_loader:
                if not isinstance(feature, torch.Tensor):
                    feature = torch.tensor(feature).float().to(self.device)

                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target).float().to(self.device)

                if feature.dtype != torch.float32:
                    feature = feature.float()

                if target.dtype != torch.float32:
                    target = target.float()

                tr_loss, _ = self.train_step(feature, target)
                train_loss.append(tr_loss.item())

            self.loss["train"].append(sum(train_loss) / len(train_loss))
            vl_loss, _ = self.evaluate(valid_loader)
            self.loss["val"].append(vl_loss)

            print(f"Epoch {epoch + 1}/{self.settings.n_epochs} | Train loss: {self.loss['train'][-1]} | Val loss: {self.loss['val'][-1]}")

            early_stopping(vl_loss, self.model, self.settings.checkpoint_path)

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        best_model_path = f"{self.settings.checkpoint_path}/checkpoint.ckpt"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def predict(
        self,
        dataloader: DataLoader,
        trained_model: nn.Module,
    ) -> Union[torch.Tensor, np.ndarray]:
        trained_model.eval()
        predictions = []

        with torch.no_grad():
            for feature, _ in dataloader:

                if not isinstance(feature, torch.Tensor):
                    feature = torch.tensor(feature).float().to(self.device)

                if feature.dtype != torch.float32:
                    feature = feature.float()

                output = trained_model(feature)
                predictions.append(output)

        return torch.cat(predictions)
