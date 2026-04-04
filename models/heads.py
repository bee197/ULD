import torch
from torch import nn
import torch.nn.functional as F


class TwoHotRewardHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_bins: int,
        vmin: float,
        vmax: float,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.linear = nn.Linear(latent_dim, num_bins)

        # 变换后的 bin 边界（在 symexp 空间内均匀分布）
        vmin_trans = self.symexp(torch.tensor(vmin)).item()
        vmax_trans = self.symexp(torch.tensor(vmax)).item()
        # 生成均匀的 bin 中心
        self.register_buffer('bin_centers_trans',
                             torch.linspace(vmin_trans, vmax_trans, num_bins))
        self.bin_spacing = (vmax_trans - vmin_trans) / (num_bins - 1)

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        """symexp(x) = sign(x) * (exp(|x|) - 1)"""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def forward(self, z_sa: torch.Tensor) -> torch.Tensor:
        """返回每个 bin 的 logits (batch_size, num_bins)"""
        return self.linear(z_sa)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch, num_bins) 未归一化预测
        target: (batch,) 原始奖励值
        返回交叉熵损失
        """
        with torch.no_grad():
            # 将目标映射到变换空间
            target_trans = self.symexp(target)
            # 裁剪到 bin 中心范围，避免超出边界
            target_trans = torch.clamp(target_trans,
                                       self.bin_centers_trans[0],
                                       self.bin_centers_trans[-1])
            # 计算在 bin 中心数组中的连续索引
            pos = (target_trans - self.bin_centers_trans[0]) / self.bin_spacing
            lower_idx = torch.floor(pos).long()
            upper_idx = torch.ceil(pos).long()
            # 边界裁剪
            lower_idx = torch.clamp(lower_idx, 0, self.num_bins - 1)
            upper_idx = torch.clamp(upper_idx, 0, self.num_bins - 1)
            # 权重：距离较近的 bin 权重更大
            upper_weight = pos - lower_idx.float()
            lower_weight = 1.0 - upper_weight

            # 构建 two‑hot 目标（软标签）
            two_hot = torch.zeros_like(logits)
            # 当上下索引相同时，只赋值一次（权重为1）
            same = (lower_idx == upper_idx)
            two_hot.scatter_(1,
                             lower_idx.unsqueeze(1),
                             torch.where(same,
                                         torch.ones_like(lower_weight).unsqueeze(1),
                                         lower_weight.unsqueeze(1)))
            two_hot.scatter_(1,
                             upper_idx.unsqueeze(1),
                             torch.where(same,
                                         torch.zeros_like(upper_weight).unsqueeze(1),
                                         upper_weight.unsqueeze(1)))

        # 使用 log_softmax 和加权求和计算交叉熵
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(two_hot * log_probs).sum(dim=-1).mean()
        return loss