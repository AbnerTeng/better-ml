from typing import Optional, Tuple, Union

import torch
from torch import nn


class RNN(nn.Module):
    """
    A simple RNN model.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden units.
        output_dim (int): The number of output units.

    Attributes:
        rnn (nn.RNN): The RNN layer.
        fc (nn.Linear): The output layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.rnn = self._get_rnn_layer()
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def _get_rnn_layer(self):
        model_map = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }
        return model_map[self.rnn_type](
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        hiddens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            hiddens (Tuple[torch.Tensor, torch.Tensor]): The hidden states of the model.

        Returns:
            output (torch.Tensor): The output tensor.
            hiddens (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): The hidden states of the model.
        """
        output, (hidden, cell) = self.rnn(x, hiddens)
        output = self.dropout(output)
        output = self.output_layer(output)

        return output, (hidden, cell)
