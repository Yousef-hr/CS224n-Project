import torch

def __covariance__(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    n = max(x.size(0), 1)
    return (x.T @ x) / n


def covariance_spectrum(x: torch.Tensor) -> torch.Tensor:
    cov = __covariance__(x)
    eigvals = torch.linalg.eigvalsh(cov).real
    return torch.sort(torch.clamp(eigvals, min=0.0), descending=True).values


def effective_rank(x: torch.Tensor, eps: float = 1e-12) -> float:
    eigvals = covariance_spectrum(x)
    total = eigvals.sum()
    if total <= eps:
        return 0.0
    probs = eigvals / (total + eps)
    entropy = -(probs * torch.log(probs + eps)).sum()
    return float(torch.exp(entropy).item())


def variance_ratio(pred_emb: torch.Tensor, target_emb: torch.Tensor, eps: float = 1e-8) -> float:
    pred_var = pred_emb.var(dim=0, unbiased=False)
    target_var = target_emb.var(dim=0, unbiased=False)
    return float((pred_var / (target_var + eps)).mean().item())