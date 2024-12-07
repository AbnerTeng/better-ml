import torch
from torch import nn


class VanillaNN(nn.Module):
    """
    Plain neural network model with 2 hidden layers.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden units.
        output_dim (int): The number of output units.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            logits (torch.Tensor): The output tensor.
        """
        output = self.model(x)
        logits = torch.sigmoid(output)

        return logits
