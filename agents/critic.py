import torch
from torch import nn

class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,      # 注意：改为 input_dim，不再固定为 latent_dim
        hidden_dims: list,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TwinCritic(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list):
        super().__init__()
        # 直接使用 latent_dim 作为输入维度（因为接收 z_sa）
        self.q1 = Critic(latent_dim, hidden_dims)
        self.q2 = Critic(latent_dim, hidden_dims)

    def forward(self, z_sa: torch.Tensor):
        """
        Args:
            z_sa: (B, latent_dim)  状态-动作嵌入

        Returns:
            q1, q2: each (B, 1)
        """
        return self.q1(z_sa), self.q2(z_sa)