import torch
from torch import nn


class Actor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list,
        log_std_min: float = -10,
        log_std_max: float = 2,
    ):
        pass

    def forward(self, z_s: torch.Tensor) -> dict:
        """
        Returns:
            {
                "action": Tensor,
                "log_prob": Tensor,
                "mean": Tensor
            }
        """
        pass

    def sample(self, z_s):
        pass