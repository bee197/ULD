import torch
from torch import nn


class LatentDynamics(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        use_linear: bool = True,
    ):
        pass

    def forward(
        self,
        z_sa: torch.Tensor
    ) -> dict:
        """
        Returns:
            {
                "z_next": Tensor,
                "reward": Tensor,
                "done": Tensor
            }
        """
        pass