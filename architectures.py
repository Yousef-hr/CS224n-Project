import torch
import torch.nn as nn

class Projector(nn.Module):
    """
    MLP projector built from a spec string like '256-512-128'.
    """

    def __init__(self, mlp_spec: str):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
