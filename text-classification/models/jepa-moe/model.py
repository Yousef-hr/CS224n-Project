"""
JEPA-style text classification: MoE predictor maps input embeddings to label space.
Encoder owned by runner.
"""

import torch

from utils.architectures import MoEMLP
from utils.nn import init_module_weights
from utils.losses import BCS_SIGReg_Loss, cosine_similarity_loss

from ...base import BaseTextClassificationModel, RunContext


class MoEJEPATextClassifier(BaseTextClassificationModel):
    def __init__(
        self,
        embed_dim: int,
        predictor_hidden_dim: int = 1024,
        moe_num_experts: int = 4,
        sigreg_weight: float = 0.1,
        sigreg_num_slices: int = 256,
        sigreg_lmbd: float = 10.0,
    ):
        super().__init__()
        self.head = MoEMLP(embed_dim=embed_dim, hidden_dim=predictor_hidden_dim, num_experts=moe_num_experts)
        self.apply(init_module_weights)

        self.sigreg_weight = float(sigreg_weight)
        self.sigreg_loss = BCS_SIGReg_Loss(num_slices=sigreg_num_slices, lmbd=sigreg_lmbd)


    def forward(self, input_embeddings: torch.Tensor, ctx: RunContext | None = None) -> torch.Tensor:
        return self.head(input_embeddings)


    def forward_with_diagnostics(
        self, input_embeddings: torch.Tensor, ctx: RunContext | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_probs = self.head.gate_probs(input_embeddings)
        expert_outputs = self.head.expert_outputs(input_embeddings)
        pred_emb = torch.sum(gate_probs.unsqueeze(-1) * expert_outputs, dim=1)
        return pred_emb, gate_probs, expert_outputs


    def loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        ctx: RunContext | None = None,
    ) -> dict[str, torch.Tensor]:
        if ctx is None or ctx.label_embeddings is None:
            raise ValueError("MoEJEPATextClassifier requires ctx.label_embeddings for loss()")

        pred_emb = outputs
        pred_norm = pred_emb / (pred_emb.norm(dim=-1, keepdim=True) + 1e-12)
        targets = ctx.label_embeddings[labels]

        alignment_loss = cosine_similarity_loss(pred_norm, targets)
        sigreg_out = self.sigreg_loss(pred_norm, targets)
        sigreg_loss = sigreg_out["loss"]

        total = alignment_loss + self.sigreg_weight * sigreg_loss
        return {
            "total_loss": total,
            "alignment_loss": alignment_loss,
            "sigreg_loss": sigreg_loss,
        }


    def scores(self, outputs: torch.Tensor, ctx: RunContext | None = None) -> torch.Tensor:
        if ctx is None or ctx.label_embeddings is None:
            raise ValueError("MoEJEPATextClassifier requires ctx.label_embeddings for scores()")

        pred_norm = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
        return pred_norm @ ctx.label_embeddings.T
