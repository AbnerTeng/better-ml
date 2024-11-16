import os

from rich import print
import torch
from torch import nn


class EarlyStopping:
    """
    Early stopping class

    Args:
        patience: int, default=7
        verbose: bool, default=False
    """
    def __init__(self, patience: int = 7, verbose: bool = False) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss: float, model: nn.Module, path: str) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score:
            self.counter += 1

            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(
        self,
        val_loss: float,
        model: nn.Module,
        path: str
    ) -> None:
        """
        Saving checkpoints if validation loss decreased

        Args:
            val_loss: float
            model: nn.Module
            path: str (checkpoint path)
        """
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(model.state_dict(), f"{path}/checkpoint.ckpt")
        self.val_loss_min = val_loss
