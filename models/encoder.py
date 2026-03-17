import torch
from torch import nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    def __init__(
        self,
        obs_shape: tuple,
        latent_dim: int,
        encoder_type: str = "mlp",   # "mlp" or "cnn"
        hidden_dims: list = [256, 256],
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.latent_dim = latent_dim

        # =========================
        # MLP encoder (vector input)
        # =========================
        if encoder_type == "mlp":

            assert len(obs_shape) == 1, "MLP expects flat vector input"

            input_dim = obs_shape[0]

            layers = []
            prev_dim = input_dim

            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.ReLU())
                prev_dim = h

            layers.append(nn.Linear(prev_dim, latent_dim))

            self.encoder = nn.Sequential(*layers)

        # =========================
        # CNN encoder (image input)
        # =========================
        elif encoder_type == "cnn":

            assert len(obs_shape) == 3, "CNN expects (C, H, W)"

            C, H, W = obs_shape

            self.conv = nn.Sequential(
                nn.Conv2d(C, 32, 3, stride=2),   # downsample
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2),
                nn.ReLU(),
            )

            # 计算 conv 输出尺寸
            with torch.no_grad():
                dummy = torch.zeros(1, C, H, W)
                conv_out_dim = self.conv(dummy).view(1, -1).shape[1]

            self.fc = nn.Sequential(
                nn.Linear(conv_out_dim, latent_dim)
            )

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # =========================
        # 可选：LayerNorm（很重要）
        # =========================
        self.norm = nn.LayerNorm(latent_dim)

    # ==========================================
    # Forward
    # ==========================================
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:
                MLP: [B, obs_dim]
                CNN: [B, C, H, W]

        Returns:
            z_s: [B, latent_dim]
        """

        if self.encoder_type == "mlp":

            z = self.encoder(obs)

        elif self.encoder_type == "cnn":

            # normalize image to [0,1]
            if obs.dtype == torch.uint8:
                obs = obs.float() / 255.0

            z = self.conv(obs)
            z = z.view(z.size(0), -1)
            z = self.fc(z)

        # =========================
        # normalization（关键）
        # =========================
        z = self.norm(z)

        return z
