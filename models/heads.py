import torch
from torch import nn


class TwoHotRewardHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_bins: int,
        vmin: float,
        vmax: float,
    ):
        pass

    def forward(self, z_sa: torch.Tensor) -> torch.Tensor:
        """logits over bins"""
        pass

    def loss(self, logits, target) -> torch.Tensor:
        pass