import torch
from torch import nn
import torch.nn.functional as F


LOG_STD_MIN = -10
LOG_STD_MAX = 2


class Actor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list,
        log_std_min: float = -10,
        log_std_max: float = 2,
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim

        # ===== backbone =====
        layers = []
        input_dim = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        self.backbone = nn.Sequential(*layers)

        # ===== mean & log_std =====
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

    def forward(self, z_s: torch.Tensor) -> dict:
        """
        Returns:
            {
                "action": Tensor,
                "log_prob": Tensor,
                "mean": Tensor
            }
        """

        h = self.backbone(z_s)

        mean = self.mean_head(h)
        log_std = self.log_std_head(h)

        # clamp log_std（非常关键）
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # ===== reparameterization =====
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # (B, action_dim)

        # ===== tanh squash =====
        action = torch.tanh(x_t)

        # ===== log_prob（带 squash correction）=====
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # tanh correction
        log_prob -= torch.sum(
            torch.log(1 - action.pow(2) + 1e-6),
            dim=-1,
            keepdim=True
        )

        return {
            "action": action,
            "log_prob": log_prob,
            "mean": torch.tanh(mean)  # 常用于 eval
        }

    def sample(self, z_s):
        """
        只返回 action（方便 rollout）
        """
        return self.forward(z_s)["action"]