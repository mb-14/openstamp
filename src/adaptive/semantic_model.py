"""Semantic Mapping Model from Adaptive Text Watermark (Liu & Bu, ICML 2024)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc(x)) + x


class SemanticModel(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 384,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
