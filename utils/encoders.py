import torch
import torch.nn as nn
import open_clip

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