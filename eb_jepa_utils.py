"""
Utilities adapted from facebookresearch/eb_jepa (https://github.com/facebookresearch/eb_jepa).
See eb_jepa_reference.md for full documentation of reused components.
"""

import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# nn_utils.py
# ---------------------------------------------------------------------------


def init_module_weights(m: nn.Module, std: float = 0.02) -> None:
    """
    Initialize weights for common layer types using truncated normal distribution.
    Adapted from eb_jepa.nn_utils.init_module_weights.

    Use via: module.apply(lambda m: init_module_weights(m, std=0.02))

    Args:
        m: PyTorch module to initialize
        std: Standard deviation for truncated normal initialization (default: 0.02)
    """
    if isinstance(
        m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
    ):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# architectures.py - Projector
# ---------------------------------------------------------------------------


class Projector(nn.Module):
    """
    MLP projector built from a spec string like '256-512-128'.
    Adapted from eb_jepa.architectures.Projector.
    """

    def __init__(self, mlp_spec: str):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# jepa.py - JEPAProbe pattern
# ---------------------------------------------------------------------------


class JEPAProbeBase(nn.Module):
    """
    Base pattern for JEPA with a trainable prediction head and frozen encoder.
    Adapted from eb_jepa.jepa.JEPAProbe.

    The encoder is kept fixed; only the head is trained.
    forward() detaches encoder output before passing to head.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module, loss_fn: callable):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.loss_fn = loss_fn

    @torch.no_grad()
    def encode(self, x: Any) -> torch.Tensor:
        """Encode inputs through the frozen encoder."""
        return self.encoder(x)

    def apply_head(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the head to embeddings (no gradient through encoder)."""
        return self.head(embeddings)

    def forward(self, x: Any, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode (detached) -> head -> loss."""
        with torch.no_grad():
            state = self.encoder(x)
        output = self.head(state.detach())
        return self.loss_fn(output, targets)


# ---------------------------------------------------------------------------
# training_utils.py
# ---------------------------------------------------------------------------


def setup_device(device: str = "auto") -> torch.device:
    """
    Set up the compute device.
    Adapted from eb_jepa.training_utils.setup_device.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    return device


def setup_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.
    Adapted from eb_jepa.training_utils.setup_seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    epoch: int = 0,
    **extra_state: Any,
) -> None:
    """
    Save a training checkpoint.
    Adapted from eb_jepa.training_utils.save_checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint.update(extra_state)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> dict:
    """
    Load a training checkpoint. Returns dict with epoch and extra_state.
    Adapted from eb_jepa.training_utils.load_checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    if state_dict is None:
        raise ValueError("Checkpoint must contain 'model_state_dict' or 'model_state'")
    model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "epoch": checkpoint.get("epoch", 0),
        **{k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "optimizer_state_dict", "epoch")},
    }
