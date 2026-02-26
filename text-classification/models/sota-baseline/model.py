"""
SOTA non-JEPA baseline: cross-entropy classifier on frozen encoder.

- Head: multi-layer MLP ending in Linear(embed_dim, num_classes).
- Loss: cross-entropy (standard supervised). Encoder is owned by the runner.
"""

import torch
import torch.nn as nn

from utils.losses import cross_entropy_loss
from utils.nn import init_module_weights

from ...base import BaseTextClassificationModel, RunContext


class SOTABaselineProbe(nn.Module):
    """
    Supervised text classifier: trainable MLP → logits.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.num_classes = num_classes

        layers = []
        dims = [embed_dim] + list(hidden_dims) + [num_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))

        self.head = nn.Sequential(*layers)
        self.apply(init_module_weights)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(input_embeddings)


class SOTABaselineTextClassifier(BaseTextClassificationModel):
    """
    Supervised baseline: trainable MLP head → logits. Encoder owned by runner.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.head = SOTABaselineProbe(
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, input_embeddings: torch.Tensor, ctx: RunContext | None = None) -> torch.Tensor:
        return self.head(input_embeddings)

    def loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        ctx: RunContext | None = None,
    ) -> dict[str, torch.Tensor]:
        ce = cross_entropy_loss(outputs, labels)
        return {
            "total_loss": ce,
            "cross_entropy_loss": ce,
        }

    def scores(self, outputs: torch.Tensor, ctx: RunContext | None = None) -> torch.Tensor:
        return outputs
