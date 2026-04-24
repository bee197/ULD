import copy
import torch
from torch import nn


class TargetNetwork:
    def __init__(self, online: nn.Module, tau: float):
        """
        Args:
            online: 要跟踪的网络
            tau: soft update 系数（通常 0.005 / 0.01）
        """
        self.online = online
        self.tau = tau

        # deep copy 一个 target 网络
        self.target = copy.deepcopy(online)

        # target 不参与梯度
        for p in self.target.parameters():
            p.requires_grad = False

    def update(self):
        """soft update: θ_target ← τ θ_online + (1-τ) θ_target"""
        with torch.no_grad():
            for p_online, p_target in zip(
                self.online.parameters(),
                self.target.parameters()
            ):
                p_target.data.mul_(1 - self.tau)
                p_target.data.add_(self.tau * p_online.data)