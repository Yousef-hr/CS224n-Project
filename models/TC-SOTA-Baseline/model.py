"""
SOTA non-JEPA baseline: cross-entropy classifier on frozen encoder.

- Encoder: same frozen Open-CLIP text transformer as JEPA (or RoBERTa-base).
- Head: multi-layer MLP ending in Linear(embed_dim, num_classes).
- Loss: cross-entropy (standard supervised). Used for fair comparison with JEPA.
"""

import torch
import torch.nn as nn

from utils.nn import init_module_weights


class SOTABaselineTextClassifier(nn.Module):
    """
    Supervised text classifier: frozen encoder + trainable MLP → logits.
    Standard baseline in MTEB-style comparisons (cross-entropy, no contrastive objective).
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.encoder = encoder
        embed_dim = getattr(encoder, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("Encoder must have an embed_dim attribute (e.g. OpenCLIPTextEncoder).")
        self.num_classes = num_classes

        # MLP: embed_dim -> hidden_dims[0] -> ... -> hidden_dims[-1] -> num_classes
        layers = []
        dims = [embed_dim] + list(hidden_dims) + [num_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))

        self.head = nn.Sequential(*layers).to(self.device)
        self.head.apply(lambda m: init_module_weights(m, std=0.02))

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text strings with frozen encoder. Returns [batch, embed_dim]."""
        return self.encoder(texts)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_embeddings: [batch, embed_dim] from encode_input.

        Returns:
            logits [batch, num_classes].
        """
        return self.head(input_embeddings)
