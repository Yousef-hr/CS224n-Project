"""
Deep JEPA VQA: frozen CLIP (image + question) -> deep residual MLP predictor
-> cosine alignment against CLIP choice embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn import init_module_weights
from vision_qa.base import RunContext
from vision_qa.models._jepa_base import VisionQAJEPABase


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class DeepJEPAPredictor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    hidden_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_proj(x)
        return F.normalize(x, dim=-1)


class VisionQAJEPADeep(VisionQAJEPABase):
    def __init__(
        self,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        hidden_dim: int = 1024,
        depth: int = 3,
        dropout: float = 0.1,
        lr: float = 3e-4,
    ):
        super().__init__(clip_model=clip_model, clip_pretrained=clip_pretrained)
        self.head = DeepJEPAPredictor(
            in_dim=self.fuse_dim,
            hidden_dim=hidden_dim,
            out_dim=self.embed_dim,
            depth=depth,
            dropout=dropout,
        )
        self.head.apply(init_module_weights)
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr)

    def train_mode(self) -> None:
        super().train_mode()
        self.head.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.head.eval()

    def forward(self, embeddings: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        return self.head(embeddings)
