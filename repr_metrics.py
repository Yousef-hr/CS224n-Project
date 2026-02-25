import math

import torch


def _covariance(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    n = max(x.size(0), 1)
    return (x.T @ x) / n


def covariance_spectrum(x: torch.Tensor) -> torch.Tensor:
    cov = _covariance(x)
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


def expert_usage_entropy(gate_probs: torch.Tensor, eps: float = 1e-12) -> tuple[float, float]:
    """
    Returns entropy and normalized entropy of average expert usage distribution.
    """
    usage = gate_probs.mean(dim=0)
    entropy = -(usage * torch.log(usage + eps)).sum()
    max_entropy = math.log(gate_probs.size(1))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else torch.tensor(0.0, device=usage.device)
    return float(entropy.item()), float(norm_entropy.item())


def conditional_routing_entropy(gate_probs: torch.Tensor, eps: float = 1e-12) -> tuple[float, float]:
    """
    Returns mean per-sample routing entropy and normalized variant.
    """
    per_sample_entropy = -(gate_probs * torch.log(gate_probs + eps)).sum(dim=1)
    entropy = per_sample_entropy.mean()
    max_entropy = math.log(gate_probs.size(1))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else torch.tensor(0.0, device=gate_probs.device)
    return float(entropy.item()), float(norm_entropy.item())


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Linear CKA between two representation matrices [N, D].
    """
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xxt = x @ x.T
    yyt = y @ y.T
    hsic_xy = (xxt * yyt).sum()
    hsic_xx = (xxt * xxt).sum().sqrt()
    hsic_yy = (yyt * yyt).sum().sqrt()
    denom = hsic_xx * hsic_yy + eps
    return float((hsic_xy / denom).item())


def expert_pairwise_cka(expert_outputs: torch.Tensor) -> tuple[float, float]:
    """
    expert_outputs: [N, E, D]
    Returns mean and max pairwise CKA across experts.
    """
    n_experts = expert_outputs.size(1)
    if n_experts < 2:
        return 0.0, 0.0

    pair_scores = []
    for i in range(n_experts):
        xi = expert_outputs[:, i, :]
        for j in range(i + 1, n_experts):
            yj = expert_outputs[:, j, :]
            pair_scores.append(linear_cka(xi, yj))

    scores = torch.tensor(pair_scores)
    return float(scores.mean().item()), float(scores.max().item())
