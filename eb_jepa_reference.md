# EB-JEPA Reference

This document lists all functions and patterns adapted from [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa) (Energy-Based Joint-Embedding Predictive Architectures) and where they are used in this project.

---

## Source Repository

- **URL**: https://github.com/facebookresearch/eb_jepa
- **Paper**: [A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2602.03604) (Terver et al., 2026)

---

## Reused Components

### 1. `init_module_weights` (eb_jepa_utils.py)

| Attribute | Value |
|-----------|-------|
| **Source** | `eb_jepa/nn_utils.py` |
| **URL** | https://github.com/facebookresearch/eb_jepa/blob/main/eb_jepa/nn_utils.py |

**Original behavior**: Initialize weights for Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d, and Linear layers using truncated normal distribution with configurable std.

**Our usage**: Initialize the predictor (Projector) weights in `model.py` before training.

**Adaptations**: None; logic is identical.

---

### 2. `Projector` (eb_jepa_utils.py)

| Attribute | Value |
|-----------|-------|
| **Source** | `eb_jepa/architectures.py` |
| **URL** | https://github.com/facebookresearch/eb_jepa/blob/main/eb_jepa/architectures.py |

**Original behavior**: MLP projector built from a spec string like `'256-512-128'`. Structure: Linear → BatchNorm → ReLU (repeated for hidden layers), final Linear with `bias=False`. Exposes `out_dim` attribute.

**Our usage**: Used as the predictor head in `JEPATextClassifier` to map Sx (input embeddings) → sỹ (predicted embeddings). Spec format: `"{embed_dim}-{predictor_hidden_dim}-{embed_dim}"`.

**Adaptations**: None; structure matches the original.

---

### 3. `JEPAProbeBase` / JEPAProbe pattern (eb_jepa_utils.py)

| Attribute | Value |
|-----------|-------|
| **Source** | `eb_jepa/jepa.py` – `JEPAProbe` class |
| **URL** | https://github.com/facebookresearch/eb_jepa/blob/main/eb_jepa/jepa.py |

**Original behavior**: JEPA with a trainable prediction head; encoder is kept fixed. `forward(observations, targets)` encodes observations with `torch.no_grad()`, applies the head to detached embeddings, and returns the loss.

**Our usage**: `JEPATextClassifier` follows this pattern:
- **Encoder**: `CLIPTextEncoder` (frozen Open-CLIP)
- **Head**: `CosineSimHead` (predictor + cosine similarity to label embeddings)
- **Forward flow**: `encode_input()` (no_grad) → `head(embeddings)` → logits → CE loss

**Adaptations**: We do not subclass `JEPAProbeBase` because our encoder takes `list[str]` (texts) instead of tensors. The structure (frozen encoder, trainable head, detached encoder output) is preserved.

---

### 4. `setup_device` (eb_jepa_utils.py)

| Attribute | Value |
|-----------|-------|
| **Source** | `eb_jepa/training_utils.py` |
| **URL** | https://github.com/facebookresearch/eb_jepa/blob/main/eb_jepa/training_utils.py |

**Original behavior**: Set up the compute device. `device="auto"` selects CUDA if available, else CPU.

**Our usage**: Used in `train.py` and `test.py` for consistent device selection.

**Adaptations**: Replaced logger with `print` to avoid `eb_jepa.logging` dependency.

---

### 5. `setup_seed` (eb_jepa_utils.py)

| Attribute | Value |
|-----------|-------|
| **Source** | `eb_jepa/training_utils.py` |
| **URL** | https://github.com/facebookresearch/eb_jepa/blob/main/eb_jepa/training_utils.py |

**Original behavior**: Set random seeds for Python, NumPy, and PyTorch (including CUDA) for reproducibility.

**Our usage**: Called at the start of `train.py` via `--seed` argument.

**Adaptations**: Replaced logger with `print`.

---

### 6. `save_checkpoint` (eb_jepa_utils.py)

| Attribute | Value |
|-----------|-------|
| **Source** | `eb_jepa/training_utils.py` |
| **URL** | https://github.com/facebookresearch/eb_jepa/blob/main/eb_jepa/training_utils.py |

**Original behavior**: Save model state dict, optional optimizer state, epoch, and extra kwargs to a checkpoint file.

**Our usage**: Used in `train.py` to save the best model (includes `eval_acc` in extra state).

**Adaptations**: Simplified; removed scheduler and scaler support. Checkpoint keys: `model_state_dict`, `optimizer_state_dict`, `epoch`, plus any extra kwargs.

---

### 7. `load_checkpoint` (eb_jepa_utils.py)

| Attribute | Value |
|-----------|-------|
| **Source** | `eb_jepa/training_utils.py` |
| **URL** | https://github.com/facebookresearch/eb_jepa/blob/main/eb_jepa/training_utils.py |

**Original behavior**: Load checkpoint and restore model (and optionally optimizer) state.

**Our usage**: Used in `test.py` to load the saved model for evaluation.

**Adaptations**: Support both `model_state_dict` and `model_state` keys for backward compatibility. Removed scheduler/scaler loading.

---

## Architecture Alignment with EB-JEPA

| EB-JEPA Concept | Our Implementation |
|-----------------|--------------------|
| Encoder | Frozen Open-CLIP text encoder |
| Action encoder | N/A (text-only, no actions) |
| Predictor | Projector MLP (embed_dim → hidden → embed_dim) |
| Regularizer | N/A (supervised; we use CE loss) |
| Prediction cost | Cosine similarity + temperature → logits → cross-entropy |
| JEPAProbe pattern | Frozen encoder + trainable head with detached encoder output |

---

## Files Modified to Use EB-JEPA Components

| File | Components Used |
|------|-----------------|
| `eb_jepa_utils.py` | All vendored utilities |
| `model.py` | `Projector`, `init_module_weights`, JEPAProbe pattern |
| `train.py` | `setup_device`, `setup_seed`, `save_checkpoint` |
| `test.py` | `setup_device`, `load_checkpoint` |
