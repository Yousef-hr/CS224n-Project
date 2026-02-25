"""
JEPA text classifier for MoE + SIGReg track.
"""

import torch
import torch.nn as nn

from utils.architectures import MoEMLP
from utils.nn import init_module_weights


class MoESigRegJEPATextClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor_hidden_dim: int = 512,
        moe_num_experts: int = 4,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.encoder = encoder
        embed_dim = getattr(encoder, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("Encoder must have an embed_dim attribute (e.g. OpenCLIPTextEncoder).")

        self.head = MoEMLP(embed_dim=embed_dim, hidden_dim=predictor_hidden_dim, num_experts=moe_num_experts).to(self.device)
        self.head.apply(lambda m: init_module_weights(m, std=0.02))

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return self.encoder(texts)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(input_embeddings)

    def forward_with_diagnostics(self, input_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_probs = self.head.gate_probs(input_embeddings)
        expert_outputs = self.head.expert_outputs(input_embeddings)
        pred_emb = self.head(input_embeddings)
        return pred_emb, gate_probs, expert_outputs
