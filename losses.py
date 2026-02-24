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


def sigreg_loss(pred_emb: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    Sigma regularization over batch statistics.
    Encourages each embedding dimension to keep sufficient standard deviation.
    """
    std = torch.sqrt(pred_emb.var(dim=0, unbiased=False) + eps)
    return torch.relu(torch.tensor(target_std, device=pred_emb.device, dtype=pred_emb.dtype) - std).mean()
