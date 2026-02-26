import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class SentenceTransformerEncoder(nn.Module):
    """Frozen sentence-transformer encoder. Encodes list of strings -> embeddings [B, D]."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self._embed_dim = self.model.config.hidden_size

        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

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
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        output = self.model(**encoded)
        embeddings = self._mean_pooling(output, encoded["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)
