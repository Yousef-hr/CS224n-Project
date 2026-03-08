"""Types and protocol for vision question-answering task (shared by train and eval)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class DataSpec:
    dataset: str
    subset: str | None = None
    cache_dir: str | Path | None = None
    batch_size: int = 32
    num_workers: int = 0

    image_size: tuple[int, int] = (224, 224)
    use_image: bool = True


@dataclass(frozen=True)
class TrainSpec:
    epochs: int = 3
    device: str = "auto"
    seed: int = 42

    save_dir: str | Path = "checkpoints"
    metrics_csv: str | Path | None = None


@dataclass(frozen=True)
class EvalSpec:
    """Eval-only settings. Batch size and num_workers come from DataSpec."""

    device: str = "auto"
    report_breakdown: bool = False  # e.g. accuracy by subject/grade for ScienceQA


@dataclass(frozen=True)
class RunContext:
    device: torch.device
    use_amp: bool
    extra: dict[str, Any] | None = None


@dataclass
class VisionQABatch:
    """
    Batched inputs for vision QA.
    - images: [B, C, H, W] (placeholder for text-only samples if needed)
    - questions: list of B strings
    - choices: list of B lists of strings (padded to max_num_choices in batch)
    - answer_indices: [B] long tensor
    - num_choices: [B] or scalar max_num_choices for masking invalid choice slots
    - subject_list / grade_list: optional for eval breakdown (e.g. ScienceQA)
    """

    images: torch.Tensor
    questions: list[str]
    choices: list[list[str]]
    answer_indices: torch.Tensor
    num_choices: torch.Tensor | None = None  # [B] actual num choices per sample; None => use full scores
    subject_list: list[str] | None = None
    grade_list: list[str] | None = None


class VisionQAModel(Protocol):
    """
    Minimal interface for the generic vision QA train/eval runners.

    - encode_inputs: (image + question + choices) -> fused embedding per example.
    - forward: embeddings -> logits [B, num_choices].
    - loss / scores for training and accuracy.
    """

    optimizer: torch.optim.Optimizer

    def encode_inputs(self, batch: VisionQABatch, ctx: RunContext) -> torch.Tensor:
        """Return fused embeddings Tensor[B, D] on ctx.device."""

    def forward(self, embeddings: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        """Return logits [B, num_choices] (or [B, max_choices] with masking)."""

    def loss(
        self,
        outputs: torch.Tensor,
        answer_indices: torch.Tensor,
        ctx: RunContext,
        batch: VisionQABatch | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return dict with at least {"total_loss": Tensor}."""

    def scores(self, outputs: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        """Return class scores Tensor[B, num_choices] for accuracy (argmax vs answer_idx)."""

    def train_mode(self) -> None:
        """Put trainable components in train mode."""

    def eval_mode(self) -> None:
        """Put components in eval mode."""
