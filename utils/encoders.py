import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
from transformers import AutoTokenizer, AutoModel


class OpenCLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder. Encodes list of strings → embeddings [B, D]."""

    def __init__(self,  name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(name)

        self.model = model
        self.tokenizer = tokenizer
        
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def embed_dim(self) -> int:
        proj = self.model.text_projection
        if proj is None:
            raise ValueError("Model has no text_projection")
        if isinstance(proj, nn.Linear):
            return proj.out_features
        return proj.shape[1]

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts)
        return self.model.encode_text(tokens)


class SentenceTransformerEncoder(nn.Module):
    """Frozen sentence-transformer encoder. Encodes list of strings -> embeddings [B, D]."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def embed_dim(self) -> int:
        return self.model.config.hidden_size

    @staticmethod
    def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        token_emb = model_output[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        return torch.sum(token_emb * mask_expanded, dim=1) / torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9
        )

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        output = self.model(**encoded)
        embeddings = self._mean_pooling(output, encoded["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)