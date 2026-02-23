"""
JEPA-style text classification model.
- Frozen Open-CLIP encoder for input text and labels
- Predictor (Projector from eb_jepa) maps input embeddings to output space
- Loss: 1 - cosine_similarity(pred_emb, target_emb) minimized directly

Follows JEPAProbe pattern from eb_jepa: frozen encoder + trainable head.
"""

import torch
import torch.nn as nn
import open_clip

from dataset import get_labels
from eb_jepa_utils import Projector, init_module_weights


def get_clip_model_and_tokenizer(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    """Load Open-CLIP model and tokenizer for text encoding."""
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder. Encodes list of strings → embeddings [B, D]."""

    def __init__(self, model, tokenizer, device: torch.device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts)
        tokens = tokens.to(self.device)
        return self.model.encode_text(tokens)


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
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        predictor_hidden_dim: int = 512,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        # Load CLIP
        model, tokenizer = get_clip_model_and_tokenizer(clip_model_name, clip_pretrained)
        self.encoder = CLIPTextEncoder(model, tokenizer, self.device).to(self.device)

        # Embedding dimension
        labels = get_labels()
        with torch.no_grad():
            enc = self.encoder(labels)
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
            emb = self.encoder(get_labels())
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


def cosine_similarity_loss(
    pred_emb: torch.Tensor, label_embeddings: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Loss = 1 - cosine_similarity(pred_emb, target_emb).
    pred_emb and label_embeddings should be L2-normalized.
    targets: [B] class indices.
    """
    target_emb = label_embeddings[targets]  # [B, D]
    cos_sim = (pred_emb * target_emb).sum(dim=-1)  # [B]
    return (1 - cos_sim).mean()
