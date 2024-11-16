from typing import Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        settings: Dict
    ) -> None:
        super().__init__(model, settings)
        self.criterion = getattr(torch.nn, settings.criterion)()
        self.optimizer = getattr(torch.optim, settings.optimizer.name)(
            self.model.parameters(),
            settings.optimizer.lr
        )

    def train_step(
        self,
        feature: torch.Tensor,
        target: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = self.model(feature).squeeze(-1)
        loss = self.criterion(preds, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, preds

    def evaluate(
        self,
        dataloader: DataLoader,
        *args,
        **kwargs
    ) -> Tuple[float, torch.Tensor]:
        self.model.eval()
        tr_loss = 0

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
