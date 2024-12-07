"""
Build PyTorch Dataset
"""

from typing import Literal, Tuple, Dict, Any

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from ..utils.utils import load_data


class HouseDataset(Dataset):
    """
    Build PyTorch Dataset for tabular regression task
    """

    def __init__(
        self,
        path: str,
        tags: Literal["train", "val", "test"],
        train_ratio: float = 0.8,
    ) -> None:
        super().__init__()
        self.path = path
        self.tags = tags
        self.train_ratio = train_ratio
        self.input_dim, self.output_dim = 0, 0
        self.feature, self.target = self.__readdata__()

    def __readdata__(
        self,
    ) -> Tuple[np.ndarray | Any | list, np.ndarray | Any | list]:
        """
        Assert file type is pickle here (tabular benchmark dataset)
        """
        data = load_data(self.path)
        feature, target = data.data, data.target

        self.input_dim, self.output_dim = (
            len(data.feature_names),
            len(data.target_names),
        )

        assert isinstance(feature, np.ndarray)
        assert isinstance(target, np.ndarray)

        train_x, test_x, train_y, test_y = train_test_split(
            feature, target, train_size=self.train_ratio
        )

        if self.tags == "train":
            return train_x, train_y

        if self.tags == "val":
            train_x, valid_x, train_y, valid_y = train_test_split(
                train_x, train_y, train_size=self.train_ratio
            )
            return valid_x, valid_y

        if self.tags == "test":
            return test_x, test_y

        else:
            raise ValueError("Invalid tag")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.feature[idx], self.target[idx]

    def __len__(self) -> int:
        return len(self.feature)


def build_dataloader(
    tags: Literal["train", "val", "test"], data_conf: Dict[str, Any]
) -> Tuple[DataLoader, int, int]:
    """
    Build PyTorch DataLoader
    """
    input_dim, output_dim = 0, 0

    dataset = HouseDataset(
        path=data_conf["path"],
        tags=tags,
        train_ratio=data_conf["train_ratio"],
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=data_conf["batch_size"],
        shuffle=data_conf["shuffle"],
        num_workers=data_conf["num_workers"],
    )

    if tags == "train":
        input_dim = dataset.input_dim
        output_dim = dataset.output_dim

    return dataloader, input_dim, output_dim
