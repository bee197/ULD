import torch
from torch import nn


class StateActionEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
    ):
        super().__init__()

        self.latent_dim = latent_dim

        input_dim = latent_dim + action_dim

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.net = nn.Sequential(*layers)

        # 🔥 很关键：LayerNorm（稳定 latent 空间）
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        z_s: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_s: [B, latent_dim]
            action: [B, action_dim]

        Returns:
            z_sa: [B, latent_dim]
        """

        # 拼接 state latent 和 action
        x = torch.cat([z_s, action], dim=-1)

        z_sa = self.net(x)

        # normalization（非常关键）
        # z_sa = self.norm(z_sa)

        return z_sa