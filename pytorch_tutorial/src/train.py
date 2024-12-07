from typing import Tuple

import torch
import omegaconf
from torch import nn
from torch.utils.data import DataLoader

from .base.base_trainer import BaseTrainer
from .utils.tools import EarlyStopping


class Trainer(BaseTrainer):
    def __init__(self, model: nn.Module, settings: omegaconf.DictConfig) -> None:
        super().__init__(model, settings)
        self.criterion = getattr(torch.nn, settings.criterion)()
        self.optimizer = getattr(torch.optim, settings.optimizer.name)(
            self.model.parameters(), settings.optimizer.lr
        )

    def train_step(
        self, feature: torch.Tensor, target: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training step for sequence prediction models

        Args:
            feature (torch.Tensor): Input tensor
            target (torch.Tensor): Target tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss and predictions
        """
        preds = self.model(feature).squeeze(-1)
        loss = self.criterion(preds, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, preds

    def evaluate(
        self, dataloader: DataLoader, *args, **kwargs
    ) -> Tuple[float, torch.Tensor]:
        self.model.eval()
        tr_loss = 0
        preds = torch.tensor([]).to(self.device)

        for feature, target in dataloader:
            if not isinstance(feature, torch.Tensor):
                feature = torch.tensor(feature).float().to(self.device)

            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target).float().to(self.device)

            if feature.dtype != torch.float32:
                feature = feature.float()

            if target.dtype != torch.float32:
                target = target.float()

            with torch.no_grad():
                preds = self.model(feature)
                loss = self.criterion(preds, target)
                tr_loss += loss.item()

        return tr_loss / len(dataloader), preds


class SequenceTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, settings: omegaconf.DictConfig) -> None:
        """
        Trainer for sequence prediction models

        Args:
            model (nn.Module): PyTorch model
            settings (omegaconf.DictConfig): Configuration settings
        """
        super().__init__(model, settings)
        self.criterion = getattr(torch.nn, settings.criterion)()
        self.optimizer = getattr(torch.optim, settings.optimizer.name)(
            self.model.parameters(), settings.optimizer.lr
        )

    def train_step(
        self,
        feature: torch.Tensor,
        target: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds, (hidden, _) = self.model(feature, (hidden, cell))
        loss = self.criterion(preds, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        hidden = hidden.detach()

        return loss, preds, hidden

    def evaluate(
        self, dataloader: DataLoader, *args, **kwargs
    ) -> Tuple[float, torch.Tensor]:
        """
        Evaluation step for sequence prediction models

        Args:
            dataloader (DataLoader): DataLoader object

        Returns:
            Tuple[float, torch.Tensor]: Loss and predictions
        """
        self.model.eval()
        tr_loss = 0
        preds = torch.tensor([]).to(self.device)
        hidden = torch.randn(self.model.num_layers, self.model.hidden_dim)
        cell = torch.randn(self.model.num_layers, self.model.hidden_dim)

        for feature, target in dataloader:
            if not isinstance(feature, torch.Tensor):
                feature = torch.tensor(feature).float().to(self.device)

            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target).float().to(self.device)

            if feature.dtype != torch.float32:
                feature = feature.float()

            if target.dtype != torch.float32:
                target = target.float()

            with torch.no_grad():
                preds, (hidden, _) = self.model(feature, (hidden, cell))
                loss = self.criterion(preds, target)
                tr_loss += loss.item()
                hidden = hidden.detach()

        return tr_loss / len(dataloader), preds

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> nn.Module:
        """
        training step for sequence prediction models
        """
        early_stopping = EarlyStopping(
            patience=self.settings.patience,
            verbose=self.settings.verbose,
        )
        for epoch in range(self.settings.n_epochs):
            train_loss = []
            self.model.train()
            hidden = torch.randn(self.model.num_layers, self.model.hidden_dim)
            cell = torch.randn(self.model.num_layers, self.model.hidden_dim)

            for feature, target in train_loader:
                if not isinstance(feature, torch.Tensor):
                    feature = torch.tensor(feature).float().to(self.device)

                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target).float().to(self.device)

                if feature.dtype != torch.float32:
                    feature = feature.float()

                if target.dtype != torch.float32:
                    target = target.float()

                tr_loss, _, hidden = self.train_step(feature, target, hidden, cell)
                train_loss.append(tr_loss.item())

            self.loss["train"].append(sum(train_loss) / len(train_loss))
            vl_loss, _ = self.evaluate(valid_loader)
            self.loss["val"].append(vl_loss)

            if early_stopping.early_stop:
                break

        best_model_path = f"{self.settings.checkpoint_path}/checkpoint.ckpt"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
