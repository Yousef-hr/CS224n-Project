"""
JEPA + SIGReg text classifier (baseline predictor + BCS SIGReg).
Encoder owned by runner.
"""

import torch
import torch.nn as nn

from utils.architectures import MLP
from utils.nn import init_module_weights
from utils.losses import BCS_SIGReg_Loss, cosine_similarity_loss

from ...base import BaseTextClassificationModel, RunContext


class SIGRegJEPATextClassifier(BaseTextClassificationModel):
    def __init__(
        self,
        embed_dim: int,
        predictor_hidden_dim: int = 1024,
        dropout: float = 0.0,
        num_slices: int = 256,
        lmbd: float = 10.0,
        sigreg_weight: float = 0.05,
    ):
        super().__init__()

        self.head = nn.Sequential(
            MLP(f"{embed_dim}-{predictor_hidden_dim}-{embed_dim}"),
            nn.Dropout(dropout),
        )
        self.apply(init_module_weights)
    
        self.sigreg_weight = float(sigreg_weight)
        self.bcs_loss = BCS_SIGReg_Loss(num_slices=num_slices, lmbd=lmbd)

    def forward(self, input_embeddings: torch.Tensor, ctx: RunContext | None = None) -> torch.Tensor:
        return self.head(input_embeddings)

    def loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        ctx: RunContext | None = None,
    ) -> dict[str, torch.Tensor]:
        if ctx is None or ctx.label_embeddings is None:
            raise ValueError("SIGRegJEPATextClassifier requires ctx.label_embeddings for loss()")
            
        pred_norm = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
        targets = ctx.label_embeddings[labels]
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

    def scores(self, outputs: torch.Tensor, ctx: RunContext | None = None) -> torch.Tensor:
        if ctx is None or ctx.label_embeddings is None:
            raise ValueError("SIGRegJEPATextClassifier requires ctx.label_embeddings for scores()")

        pred_norm = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
        return pred_norm @ ctx.label_embeddings.T
