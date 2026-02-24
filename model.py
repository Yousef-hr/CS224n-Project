"""
JEPA-style text classification model.
- Frozen Open-CLIP encoder for input text and labels
- Predictor head maps input embeddings to output space
- Loss: 1 - cosine_similarity(pred_emb, target_emb) minimized directly

Follows JEPAProbe pattern from eb_jepa: frozen encoder + trainable head.
"""

import torch
import torch.nn as nn

from architectures import Projector
from utils.nn import init_module_weights
from encoders.OpenCLIP import OpenCLIPTextEncoder, get_clip_model_and_tokenizer

class BaselinePredictorHead(nn.Module):
    """Single-projector baseline head."""

    def __init__(self, predictor: nn.Module, label_embeddings: torch.Tensor, dropout_p: float = 0.0):
        super().__init__()
        self.predictor = predictor
        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer("label_embeddings", label_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred_emb = self.predictor(self.dropout(x))
        return pred_emb / pred_emb.norm(dim=-1, keepdim=True)

class MoEPredictor(nn.Module):
    """
    Simple dense Mixture-of-Experts predictor.
    Uses a learned gate over multiple projector experts and returns weighted sum.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"MoE requires num_experts >= 2, got {num_experts}")

        self.gate = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList(
            [Projector(f"{embed_dim}-{hidden_dim}-{embed_dim}") for _ in range(num_experts)]
        )

        self.gate.apply(lambda m: init_module_weights(m, std=0.02))
        for expert in self.experts:
            expert.apply(lambda m: init_module_weights(m, std=0.02))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_probs = torch.softmax(self.gate(x), dim=-1)  # [B, E]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, E, D]
        mixed = torch.sum(gate_probs.unsqueeze(-1) * expert_outputs, dim=1)  # [B, D]
        return mixed

class MoEPredictorHead(nn.Module):
    """MoE head with shared frozen label embeddings."""

    def __init__(self, predictor: nn.Module, label_embeddings: torch.Tensor):
        super().__init__()
        self.predictor = predictor
        self.register_buffer("label_embeddings", label_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred_emb = self.predictor(x)
        return pred_emb / pred_emb.norm(dim=-1, keepdim=True)

class JEPATextClassifier(nn.Module):
    """
    JEPA-inspired text classifier with selectable prediction head:
    - Frozen encoder (CLIP) → Sx
    - Trainable head:
      - baseline: single Projector
      - moe: gated mixture of Projector experts
    """

    def __init__(
        self,
        labels: list[str],
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        head_type: str = "baseline",
        predictor_hidden_dim: int = 512,
        baseline_dropout: float = 0.0,
        moe_num_experts: int = 4,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.head_type = head_type

        # Load CLIP
        model, tokenizer = get_clip_model_and_tokenizer(clip_model_name, clip_pretrained)
        self.encoder = OpenCLIPTextEncoder(model, tokenizer, self.device).to(self.device)

        # Embedding dimension
        self.labels = labels
        with torch.no_grad():
            enc = self.encoder(self.labels)
        embed_dim = enc.shape[-1]

        # Label embeddings (frozen)
        label_emb = self._compute_label_embeddings()

        if head_type == "baseline":
            predictor = Projector(f"{embed_dim}-{predictor_hidden_dim}-{embed_dim}")
            predictor.apply(lambda m: init_module_weights(m, std=0.02))
            self.head = BaselinePredictorHead(
                predictor,
                label_emb,
                dropout_p=baseline_dropout,
            ).to(self.device)
        elif head_type == "moe":
            predictor = MoEPredictor(
                embed_dim=embed_dim,
                hidden_dim=predictor_hidden_dim,
                num_experts=moe_num_experts,
            )
            self.head = MoEPredictorHead(predictor, label_emb).to(self.device)
        else:
            raise ValueError(f"Unknown head_type '{head_type}'. Use 'baseline' or 'moe'.")

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


