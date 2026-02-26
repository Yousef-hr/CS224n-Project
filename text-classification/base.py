"""Types and protocol for text classification task (shared by train and eval)."""

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DataSpec:
    dataset: str
    subset: str | None = None
    cache_dir: str | Path | None = None
    batch_size: int = 128
    num_workers: int = 0

    embedding_cache_dir: str | Path | None = None
    precompute_batch_size: int = 512

    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"


@dataclass(frozen=True)
class TrainSpec:
    epochs: int = 3
    device: str = "auto"
    seed: int = 42
    lr: float = 3e-4
    weight_decay: float = 0.0

    save_dir: str | Path = "checkpoints"
    metrics_csv: str | Path | None = None


@dataclass(frozen=True)
class EvalSpec:
    """Eval-only settings. Batch size and num_workers come from DataSpec."""

    device: str = "auto"
    report_repr_metrics: bool = False
    repr_topk_eigs: int = 10


@dataclass(frozen=True)
class RunContext:
    device: torch.device
    use_amp: bool
    label_embeddings: torch.Tensor | None
    labels_list: list[str] | None = None
    extra: dict[str, Any] | None = None


class TextClassificationModel(Protocol):
    """
    Minimal interface for the generic train/eval runners.
    Runner owns encoding; models only see embeddings. forward/loss/scores use input_embeddings.
    """

    def forward(self, input_embeddings: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        """
        Return either logits [B, C] (logit-native) or predicted embeddings [B, D] (JEPA).
        """

    def loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        ctx: RunContext,
    ) -> dict[str, torch.Tensor]:
        """Return dict with at least {"total_loss": Tensor}."""

    def scores(self, outputs: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        """Return class scores Tensor[B, C] for accuracy."""

    def train_mode(self) -> None:
        """Put trainable components in train mode."""

    def eval_mode(self) -> None:
        """Put components in eval mode."""


class BaseTextClassificationModel(nn.Module):
    """
    Head-only base for text classification. Runner owns encoder; subclasses implement
    forward(input_embeddings, ctx), loss(outputs, labels, ctx), scores(outputs, ctx).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, input_embeddings: torch.Tensor, ctx: RunContext | None = None) -> torch.Tensor:
        ...

    @abstractmethod
    def loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        ctx: RunContext | None = None,
    ) -> dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def scores(self, outputs: torch.Tensor, ctx: RunContext | None = None) -> torch.Tensor:
        ...
