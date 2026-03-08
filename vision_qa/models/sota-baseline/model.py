"""
Vision QA baseline: frozen CLIP (image + text) -> fused embedding -> MLP -> logits over choices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from vision_qa.base import RunContext, VisionQABatch, VisionQAModel


MAX_CHOICES = 5  # ScienceQA has 2-5 choices; pad to 5


class VisionQABaseline(nn.Module):
    """
    Frozen CLIP image + text encoders; concat embeddings; MLP -> logits over max_choices.
    """

    def __init__(
        self,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        hidden_dim: int = 512,
        max_choices: int = MAX_CHOICES,
        dropout: float = 0.1,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained
        self.max_choices = max_choices

        model, self._preprocess_train, self._preprocess_eval = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False

        self.embed_dim = self.model.visual.output_dim  # same as text in CLIP
        fuse_dim = self.embed_dim * 2  # concat image + text
        self.head = nn.Sequential(
            nn.Linear(fuse_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_choices),
        )
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr)

    def get_image_transform(self):
        """Return the CLIP image preprocess for use in data collate."""
        return self._preprocess_eval

    def train_mode(self) -> None:
        self.head.train()
        self.model.eval()

    def eval_mode(self) -> None:
        self.head.eval()
        self.model.eval()

    def _encode_images(self, images: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        """images [B, C, H, W] -> [B, D]."""
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda" if ctx.use_amp else "cpu", enabled=ctx.use_amp
            ):
                out = self.model.encode_image(images)
        return out.to(device=ctx.device, dtype=torch.float32)

    def _encode_texts(self, texts: list[str], ctx: RunContext) -> torch.Tensor:
        """List of B strings -> [B, D]."""
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda" if ctx.use_amp else "cpu", enabled=ctx.use_amp
            ):
                tokens = self.tokenizer(texts)
                tokens = tokens.to(ctx.device)
                out = self.model.encode_text(tokens)
        return out.to(device=ctx.device, dtype=torch.float32)

    def encode_inputs(self, batch: VisionQABatch, ctx: RunContext) -> torch.Tensor:
        """Fuse image and text (question + choices) into [B, 2*D]."""
        img_emb = self._encode_images(batch.images, ctx)  # [B, D]
        # Text: "Question: ... Choices: (0) ... (1) ..."
        texts = []
        for q, choices in zip(batch.questions, batch.choices):
            parts = [f"Question: {q}", "Choices:"]
            for i, c in enumerate(choices):
                if c:
                    parts.append(f"({i}) {c}")
            texts.append(" ".join(parts))
        text_emb = self._encode_texts(texts, ctx)  # [B, D]
        fused = torch.cat([img_emb, text_emb], dim=1)  # [B, 2*D]
        return fused

    def forward(self, embeddings: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        """[B, fuse_dim] -> [B, max_choices]."""
        return self.head(embeddings)

    def loss(
        self,
        outputs: torch.Tensor,
        answer_indices: torch.Tensor,
        ctx: RunContext,
        batch: VisionQABatch | None = None,
    ) -> dict[str, torch.Tensor]:
        """Cross-entropy; mask invalid choice slots when num_choices is provided."""
        logits = outputs
        if batch is not None and batch.num_choices is not None:
            n = logits.size(1)
            num_c = batch.num_choices.to(logits.device)
            mask = torch.arange(n, device=logits.device, dtype=torch.long) < num_c.unsqueeze(1)
            logits = logits.masked_fill(~mask, -1e9)
        ce = F.cross_entropy(logits, answer_indices)
        return {"total_loss": ce, "cross_entropy_loss": ce}

    def scores(self, outputs: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        return outputs
