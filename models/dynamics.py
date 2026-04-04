import torch
from torch import nn


class LatentDynamics(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        use_linear: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # === 核心：论文里的 linear environment model ===
        if use_linear:
            # 一次性预测 [z_next, reward, done]
            self.model = nn.Linear(latent_dim, latent_dim + 2)
        else:
            # 非论文版本（扩展）
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim + 2),
            )

    def forward(
        self,
        z_sa: torch.Tensor
    ) -> dict:
        """
        Args:
            z_sa: (B, latent_dim)

        Returns:
            {
                "z_next": (B, latent_dim),
                "reward": (B, 1),
                "done": (B, 1)
            }
        """

        out = self.model(z_sa)

        # 拆分输出
        z_next = out[:, :self.latent_dim]
        reward = out[:, self.latent_dim:self.latent_dim + 1]
        done = out[:, self.latent_dim + 1:]

        return {
            "z_next": z_next,
            "reward": reward,
            "done": done
        }