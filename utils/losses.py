import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Loss = 1 - cosine_similarity(predictions, targets).
    """
    cos_sim = (predictions * targets).sum(dim=-1)  # [B]
    return (1 - cos_sim).mean()


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy for SOTA baseline classifier. logits [B, C], labels [B] long."""
    return F.cross_entropy(logits, labels)


def all_reduce(x, op):
    """All-reduce operation for distributed training."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x, op=op)
        return x
    else:
        return x


def epps_pulley(x, t_min=-3, t_max=3, n_points=10):
    """Epps-Pulley test statistic for Gaussianity."""
    # integration points
    t = torch.linspace(t_min, t_max, n_points, device=x.device)
    # theoretical CF for N(0, 1)
    exp_f = torch.exp(-0.5 * t**2)
    # ECF
    x_t = x.unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    # weighted L2 distance
    err = exp_f * (ecf - exp_f).abs() ** 2
    T = torch.trapz(err, t, dim=1)
    return T


class BCS_SIGReg_Loss(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg."""

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd

    def forward(self, z1, z2):
        with torch.no_grad():
            dev = z1.device
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A /= A.norm(p=2, dim=0)

        view1 = z1 @ A
        view2 = z2 @ A

        self.step += 1
        bcs = (epps_pulley(view1).mean() + epps_pulley(view2).mean()) / 2
        invariance_loss = F.mse_loss(z1, z2).mean()
        total_loss = invariance_loss + self.lmbd * bcs
        return {"loss": total_loss, "bcs_loss": bcs, "invariance_loss": invariance_loss}