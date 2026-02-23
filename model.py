"""
JEPA-style text classification model.
- Frozen Open-CLIP encoder for input text and labels
- Predictor (Projector from eb_jepa) maps input embeddings to output space
- Loss: 1 - cosine_similarity(pred_emb, target_emb) minimized directly

Follows JEPAProbe pattern from eb_jepa: frozen encoder + trainable head.
"""

import torch
import torch.nn as nn

from architectures import Projector
from utils.nn import init_module_weights
from encoders.OpenCLIP import OpenCLIPTextEncoder, get_OpenCLIP_model_and_tokenizer

class PredictorHead(nn.Module):
    """Predictor that maps Sx → pred_emb (L2-normalized). Holds label_embeddings for loss/inference."""

    def __init__(self, predictor: nn.Module, label_embeddings: torch.Tensor):
        super().__init__()
        self.predictor = predictor
        self.register_buffer("label_embeddings", label_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred_emb = self.predictor(x)
        return pred_emb / pred_emb.norm(dim=-1, keepdim=True)

class JEPATextClassifier(nn.Module):
    """
    JEPA-inspired text classifier following eb_jepa JEPAProbe pattern:
    - Frozen encoder (CLIP) → Sx
    - Trainable head (Projector) → pred_emb (normalized)
    """

    def __init__(
        self,
        labels: list[str],
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        predictor_hidden_dim: int = 512,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        # Load CLIP
        model, tokenizer = get_OpenCLIP_model_and_tokenizer(clip_model_name, clip_pretrained)
        self.encoder = OpenCLIPTextEncoder(model, tokenizer, self.device).to(self.device)

        # Embedding dimension
        self.labels = labels
        with torch.no_grad():
            enc = self.encoder(self.labels)
        embed_dim = enc.shape[-1]

        # Predictor: Projector from eb_jepa (MLP spec)
        predictor = Projector(f"{embed_dim}-{predictor_hidden_dim}-{embed_dim}")
        predictor.apply(lambda m: init_module_weights(m, std=0.02))

        # Label embeddings (frozen)
        label_emb = self._compute_label_embeddings()

        self.head = PredictorHead(predictor, label_emb).to(self.device)

    def _compute_label_embeddings(self) -> torch.Tensor:
        """Encode label strings with frozen CLIP and L2-normalize."""
        with torch.no_grad():
            emb = self.encoder(self.labels)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def encode_input(self, texts: list[str]) -> torch.Tensor:
        """Encode input texts with frozen CLIP. Returns Sx [batch, embed_dim]."""
        return self.encoder(texts)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_embeddings: Sx from encode_input, [batch, embed_dim]

        Returns:
            pred_emb [batch, embed_dim] (L2-normalized)
        """
        return self.head(input_embeddings)


