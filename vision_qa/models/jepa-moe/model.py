"""
JEPA MoE VQA: frozen CLIP (image + question) → linear projection → Mixture-of-
Experts predictor → cosine alignment against CLIP choice embeddings.

A learned linear projection maps the fused 2D input down to D so that MoEMLP
(which requires input_dim == output_dim) can operate in the CLIP embedding space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.architectures import MoEMLP
from utils.nn import init_module_weights

from vision_qa.base import RunContext
from vision_qa.models._jepa_base import VisionQAJEPABase


class VisionQAJEPAMoE(VisionQAJEPABase):
    def __init__(
        self,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        hidden_dim: int = 1024,
        moe_num_experts: int = 4,
        lr: float = 3e-4,
    ):
        super().__init__(clip_model=clip_model, clip_pretrained=clip_pretrained)
        self.proj = nn.Linear(self.fuse_dim, self.embed_dim)
        self.head = MoEMLP(
            embed_dim=self.embed_dim,
            hidden_dim=hidden_dim,
            num_experts=moe_num_experts,
        )
        self.proj.apply(init_module_weights)
        self.head.apply(init_module_weights)
        self.optimizer = torch.optim.AdamW(
            list(self.proj.parameters()) + list(self.head.parameters()), lr=lr
        )

    def train_mode(self) -> None:
        super().train_mode()
        self.proj.train()
        self.head.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.proj.eval()
        self.head.eval()

    def forward(self, embeddings: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        return self.head(self.proj(embeddings))

    def forward_with_diagnostics(
        self, embeddings: torch.Tensor, ctx: RunContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.proj(embeddings)
        gate_probs = self.head.gate_probs(x)
        expert_outputs = self.head.expert_outputs(x)
        pred_emb = torch.sum(gate_probs.unsqueeze(-1) * expert_outputs, dim=1)
        return pred_emb, gate_probs, expert_outputs
