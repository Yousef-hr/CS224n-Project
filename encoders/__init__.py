"""Encoder factory shared across all model train scripts."""

import torch


def build_encoder(
    encoder_type: str,
    device: torch.device,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "laion2b_s34b_b79k",
    st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Returns (encoder, cache_model_id, cache_pretrained_id).
    cache_model_id / cache_pretrained_id are used as keys for the embedding cache.
    """
    if encoder_type == "minilm":
        from encoders.SentenceTransformer import SentenceTransformerEncoder

        encoder = SentenceTransformerEncoder(model_name=st_model_name, device=device)
        return encoder, "all-MiniLM-L6-v2", "sentence-transformers"
    else:
        from encoders.OpenCLIP import OpenCLIPTextEncoder

        encoder = OpenCLIPTextEncoder(
            clip_model_name=clip_model,
            clip_pretrained=clip_pretrained,
            device=device,
        )
        return encoder, clip_model, clip_pretrained
