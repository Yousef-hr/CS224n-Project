"""
JEPA-style text classification: predictor head maps input embeddings to label space.
Encoder owned by runner.
"""

import torch
import torch.nn as nn

from utils.architectures import MLP
from utils.losses import cosine_similarity_loss
from utils.nn import init_module_weights

from ...base import BaseTextClassificationModel, RunContext


class BaselineJEPATextClassifier(BaseTextClassificationModel):
    def __init__(
        self,
        embed_dim: int,
        predictor_hidden_dim: int = 512,
        baseline_dropout: float = 0.0,
    ):
        super().__init__()
        self.head = nn.Sequential(
            MLP(f"{embed_dim}-{predictor_hidden_dim}-{embed_dim}"),
            nn.Dropout(baseline_dropout),
        )
        self.apply(init_module_weights)

    def forward(self, input_embeddings: torch.Tensor, ctx: RunContext | None = None) -> torch.Tensor:
        return self.head(input_embeddings)

    def loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        ctx: RunContext | None = None,
    ) -> dict[str, torch.Tensor]:
        if ctx is None or ctx.label_embeddings is None:
            raise ValueError("BaselineJEPATextClassifier requires ctx.label_embeddings for loss()")
            
        pred_norm = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
        targets = ctx.label_embeddings[labels]
        alignment_loss = cosine_similarity_loss(pred_norm, targets)
        
        return {
            "total_loss": alignment_loss,
            "alignment_loss": alignment_loss,
        }

    def scores(self, outputs: torch.Tensor, ctx: RunContext | None = None) -> torch.Tensor:
        if ctx is None or ctx.label_embeddings is None:
            raise ValueError("BaselineJEPATextClassifier requires ctx.label_embeddings for scores()")

        pred_norm = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
        return pred_norm @ ctx.label_embeddings.T
