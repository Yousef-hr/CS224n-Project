import torch

def cosine_similarity_loss(pred_emb: torch.Tensor, label_embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Loss = 1 - cosine_similarity(pred_emb, target_emb).
    pred_emb and label_embeddings should be L2-normalized.
    targets: [B] class indices.
    """
    target_emb = label_embeddings[targets]  # [B, D]
    cos_sim = (pred_emb * target_emb).sum(dim=-1)  # [B]
    return (1 - cos_sim).mean()
