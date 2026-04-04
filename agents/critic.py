import torch
from torch import nn


class Critic(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list,
    ):
        super().__init__()

        layers = []
        input_dim = latent_dim

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        # 输出 Q-value（标量）
        layers.append(nn.Linear(input_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, z_sa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_sa: (B, latent_dim)

        Returns:
            Q: (B, 1)
        """
        return self.model(z_sa)


class TwinCritic(nn.Module):
    def __init__(self, latent_dim, action_dim: int, hidden_dims):
        super().__init__()
        input_dim = latent_dim + action_dim
        self.q1 = Critic(input_dim, hidden_dims)
        self.q2 = Critic(input_dim, hidden_dims)

    def forward(self, z_s, action):
        """
        Returns:
            q1, q2: each (B, 1)
        """
        x = torch.cat([z_s, action], dim=-1)
        return self.q1(x), self.q2(x)