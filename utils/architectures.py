import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    MLP projector built from a spec string like '256-512-128'.
    """

    def __init__(self, mlp_spec: str):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEMLP(nn.Module):
    """
    Mixture-of-Experts MLP: learned gate over multiple MLP experts, returns weighted sum.
    Each expert has hidden_dim // num_experts units (capacity-efficient: total expert
    params ~ one MLP with hidden_dim). No internal weight init.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"MoE requires num_experts >= 2, got {num_experts}")

        expert_hidden = max(1, hidden_dim // num_experts)
        self.gate = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList(
            [MLP(f"{embed_dim}-{expert_hidden}-{embed_dim}") for _ in range(num_experts)]
        )

    def gate_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(x), dim=-1) # [B, E]

    def expert_outputs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(x) for expert in self.experts], dim=1) # [B, E, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.gate_probs(x).unsqueeze(-1) * self.expert_outputs(x), dim=1)  # [B, D]
