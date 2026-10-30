"""Paired CLIP-embedding fusion heads for Fakeddit fake-news classification."""

from __future__ import annotations

import torch
from torch import nn


class ConcatFusionHead(nn.Module):
    def __init__(
        self, embedding_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((image, text), dim=-1))


class GatedFusionHead(nn.Module):
    def __init__(
        self, embedding_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.image_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, hidden_dim), nn.GELU()
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, hidden_dim), nn.GELU()
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        image_projected = self.image_projection(image)
        text_projected = self.text_projection(text)
        gate = self.gate(torch.cat((image_projected, text_projected), dim=-1))
        return self.classifier(gate * text_projected + (1.0 - gate) * image_projected)
