import torch
from torch import nn


class Critic(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list,
    ):
        pass

    def forward(self, z_sa: torch.Tensor) -> torch.Tensor:
        pass


class TwinCritic(nn.Module):
    def __init__(self, latent_dim, hidden_dims):
        self.q1 = Critic(...)
        self.q2 = Critic(...)

    def forward(self, z_sa):
        return self.q1(z_sa), self.q2(z_sa)