import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def setup_device(device: str = "auto") -> torch.device:
    """
    Set up the compute device.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    return device


def setup_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.
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
