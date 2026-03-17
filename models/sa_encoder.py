import torch
from torch import nn


class StateActionEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
    ):
        pass

    def forward(
        self,
        z_s: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """(z_s, a) -> z_sa"""
        pass