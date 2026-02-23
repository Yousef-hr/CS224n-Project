
import torch
import torch.nn as nn
import open_clip

def get_clip_model_and_tokenizer(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    """Load Open-CLIP model and tokenizer for text encoding."""
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer

class OpenCLIPTextEncoder(nn.Module):
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