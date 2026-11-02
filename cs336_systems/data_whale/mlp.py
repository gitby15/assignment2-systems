
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    # 每一层，是Linear -> Gelu
    def __init__(self, dim:int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = F.gelu(x)
        return x


def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    model = MLP(dim, num_layers).to('cuda')
    x = torch.randn(batch_size, dim, device='cuda')
    def run():
        for _ in range(num_steps):
            y = model(x).mean()
            y.backward()
    return run