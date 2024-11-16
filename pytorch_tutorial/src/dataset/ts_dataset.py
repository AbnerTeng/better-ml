"""
Build PyTorch Dataset
"""
from typing import Dict

import numpy as np
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """
    Stock dataset

    Args:
        features (np.ndarray): features (price, volume, etc.)
        targets (np.ndarray): targets (return, sharpe ratio, etc.)
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        super().__init__()
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            'features': self.features[idx],
            'targets': self.targets[idx]
        }
