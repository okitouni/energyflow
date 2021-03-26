from typing import List

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float, output_dims: List[int], out: int = 1,
                 batch_norm_layer: int = None, sigmoid: bool = False):
        super().__init__()
        layers: List[nn.Module] = []

        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, out))
        if batch_norm_layer is not None:
            if batch_norm_layer >= len(output_dims):
                raise ValueError(
                    f'input batch_norm_layer was {batch_norm_layer} when there are only {len(output_dims)} layers')
            layers.insert((batch_norm_layer+1)*2,
                          torch.nn.BatchNorm1d(output_dims[batch_norm_layer]))

        if sigmoid:
            layers.append(nn.Sigmoid())

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        out = self.layers(data)
        return out
