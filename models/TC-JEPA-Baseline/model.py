"""
JEPA-style text classification model.
- Frozen Open-CLIP encoder for input text and labels
- Predictor head maps input embeddings to output space
"""

import torch
import torch.nn as nn

from utils.architectures import MLP
from utils.nn import init_module_weights

class BaselineJEPATextClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor_hidden_dim: int = 512,
        baseline_dropout: float = 0.0,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.encoder = encoder
        embed_dim = getattr(encoder, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("Encoder must have an embed_dim attribute (e.g. OpenCLIPTextEncoder).")

        self.head = nn.Sequential(
            MLP(f"{embed_dim}-{predictor_hidden_dim}-{embed_dim}"),
            nn.Dropout(baseline_dropout),
        ).to(self.device)

        self.head.apply(lambda m: init_module_weights(m, std=0.02))

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text strings with frozen CLIP. Returns Sx [batch, embed_dim]."""
        return self.encoder(texts)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_embeddings: Sx from encode_input, [batch, embed_dim]

        Returns:
            pred_emb [batch, embed_dim] (L2-normalized)
        """
        return self.head(input_embeddings)