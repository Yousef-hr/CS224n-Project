"""
Shared base for JEPA vision QA models.

Frozen CLIP encodes images, questions, and answer choices.  A trainable
predictor head (defined by each subclass) maps fused (image, question)
embeddings into the CLIP text embedding space.  At inference the predicted
embedding is scored against every choice embedding via cosine similarity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import open_clip

from vision_qa.base import RunContext, VisionQABatch
from utils.losses import cosine_similarity_loss


class VisionQAJEPABase(nn.Module):
    """
    Subclasses must:
      1. Call ``super().__init__(clip_model, clip_pretrained)``
      2. Create ``self.head`` (nn.Module) and ``self.optimizer``
      3. Override ``forward()`` to run the head
      4. Optionally override ``loss()`` to add regularisation (SigReg)
      5. Optionally extend ``train_mode()`` / ``eval_mode()``
    """

    def __init__(
        self,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
    ):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        self.clip = model
        for p in self.clip.parameters():
            p.requires_grad = False

        self.embed_dim: int = self.clip.visual.output_dim
        self.fuse_dim: int = self.embed_dim * 2
        self._preprocess = preprocess

        self._choice_embs: torch.Tensor | None = None
        self._num_choices: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def get_image_transform(self):
        return self._preprocess

    def release_encoder(self) -> None:
        """Free the frozen CLIP encoder to save GPU memory (call after precomputation)."""
        self.clip = None
        self.tokenizer = None
        self._preprocess = None

    def train_mode(self) -> None:
        if self.clip is not None:
            self.clip.eval()

    def eval_mode(self) -> None:
        if self.clip is not None:
            self.clip.eval()

    # ------------------------------------------------------------------
    # frozen CLIP encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        self.clip.eval()
        with torch.amp.autocast(
            device_type="cuda" if ctx.use_amp else "cpu", enabled=ctx.use_amp
        ):
            out = self.clip.encode_image(images)
        return out.to(device=ctx.device, dtype=torch.float32)

    @torch.no_grad()
    def _encode_texts(self, texts: list[str], ctx: RunContext) -> torch.Tensor:
        self.clip.eval()
        with torch.amp.autocast(
            device_type="cuda" if ctx.use_amp else "cpu", enabled=ctx.use_amp
        ):
            tokens = self.tokenizer(texts).to(ctx.device)
            out = self.clip.encode_text(tokens)
        return out.to(device=ctx.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # VisionQAModel protocol
    # ------------------------------------------------------------------

    def encode_inputs(self, batch: VisionQABatch, ctx: RunContext) -> torch.Tensor:
        """Fuse image + question → [B, 2D] and cache choice embeddings."""
        img_emb = self._encode_images(batch.images, ctx)
        q_emb = self._encode_texts(batch.questions, ctx)
        fused = torch.cat([img_emb, q_emb], dim=1)

        flat_choices = [c if c else " " for choices in batch.choices for c in choices]
        counts = [len(choices) for choices in batch.choices]
        max_c = max(counts)

        all_embs = self._encode_texts(flat_choices, ctx)
        all_embs = all_embs / (all_embs.norm(dim=-1, keepdim=True) + 1e-12)

        B = len(batch.choices)
        choice_embs = torch.zeros(B, max_c, self.embed_dim, device=ctx.device)
        offset = 0
        for i, count in enumerate(counts):
            choice_embs[i, :count] = all_embs[offset : offset + count]
            offset += count

        self._choice_embs = choice_embs
        self._num_choices = (
            batch.num_choices.to(ctx.device) if batch.num_choices is not None else None
        )
        return fused

    def forward(self, embeddings: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        raise NotImplementedError

    def _pred_norm(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)

    def loss(
        self,
        outputs: torch.Tensor,
        answer_indices: torch.Tensor,
        ctx: RunContext,
        batch: VisionQABatch | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_norm = self._pred_norm(outputs)
        targets = self._choice_embs[
            torch.arange(len(answer_indices), device=ctx.device), answer_indices
        ]
        alignment_loss = cosine_similarity_loss(pred_norm, targets)
        return {"total_loss": alignment_loss, "alignment_loss": alignment_loss}

    def scores(self, outputs: torch.Tensor, ctx: RunContext) -> torch.Tensor:
        pred_norm = self._pred_norm(outputs)
        sim = torch.bmm(
            pred_norm.unsqueeze(1), self._choice_embs.transpose(1, 2)
        ).squeeze(1)
        if self._num_choices is not None:
            max_c = sim.size(1)
            mask = torch.arange(max_c, device=sim.device) < self._num_choices.unsqueeze(1)
            sim = sim.masked_fill(~mask, torch.finfo(sim.dtype).min)
        return sim
