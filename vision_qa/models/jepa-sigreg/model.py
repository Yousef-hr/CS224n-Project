"""
JEPA + SigReg VQA: frozen CLIP (image + question) → MLP predictor → cosine
alignment + BCS Gaussianity regulariser against CLIP choice embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.architectures import MLP
from utils.nn import init_module_weights
from utils.losses import BCS_SIGReg_Loss, cosine_similarity_loss

from vision_qa.base import RunContext, VisionQABatch
from vision_qa.models._jepa_base import VisionQAJEPABase


class VisionQAJEPASigReg(VisionQAJEPABase):
    def __init__(
        self,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        lr: float = 3e-4,
        num_slices: int = 256,
        lmbd: float = 10.0,
        sigreg_weight: float = 0.10,
    ):
        super().__init__(clip_model=clip_model, clip_pretrained=clip_pretrained)
        self.head = nn.Sequential(
            MLP(f"{self.fuse_dim}-{hidden_dim}-{self.embed_dim}"),
            nn.Dropout(dropout),
        )
        self.head.apply(init_module_weights)

        self.sigreg_weight = float(sigreg_weight)
        self.bcs_loss = BCS_SIGReg_Loss(num_slices=num_slices, lmbd=lmbd)
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr)

    def train_mode(self) -> None:
        super().train_mode()
        self.head.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.head.eval()

    def forward(self, embeddings: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        return self.head(embeddings)

    def loss(
        self,
        outputs: torch.Tensor,
        answer_indices: torch.Tensor,
        ctx: RunContext,
        batch: VisionQABatch | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_norm = self._pred_norm(outputs)
        targets = self._choice_embs[
            torch.arange(len(answer_indices), device=ctx.device), answer_indices
        ]
        alignment_loss = cosine_similarity_loss(pred_norm, targets)
        sigreg_out = self.bcs_loss(pred_norm, targets)
        sigreg_loss = sigreg_out["loss"]
        total = alignment_loss + self.sigreg_weight * sigreg_loss

        return {
            "total_loss": total,
            "alignment_loss": alignment_loss,
            "sigreg_loss": sigreg_loss,
            "bcs_loss": sigreg_out["bcs_loss"],
            "invariance_loss": sigreg_out["invariance_loss"],
        }
