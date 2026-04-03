"""
JEPA Baseline VQA: frozen CLIP (image + question) → MLP predictor → cosine
alignment against CLIP choice embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.architectures import MLP
from utils.nn import init_module_weights

from vision_qa.base import RunContext
from vision_qa.models._jepa_base import VisionQAJEPABase


class VisionQAJEPABaseline(VisionQAJEPABase):
    def __init__(
        self,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        lr: float = 3e-4,
    ):
        super().__init__(clip_model=clip_model, clip_pretrained=clip_pretrained)
        self.head = nn.Sequential(
            MLP(f"{self.fuse_dim}-{hidden_dim}-{self.embed_dim}"),
            nn.Dropout(dropout),
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
