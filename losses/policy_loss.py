import torch
from typing import Dict


class PolicyLoss:
    """
    策略损失类
    实现 ULD 论文中的确定性策略梯度损失：
    L_policy = -0.5 * (Q1(z_sa) + Q2(z_sa)) + λ_pre * u_π²
    """

    def __init__(self, lambda_pre: float = 0.0):
        """
        参数:
            lambda_pre: 预激活正则化系数（论文中用于避免稀疏奖励环境中的局部最优）
        """
        self.lambda_pre = lambda_pre

    def compute(
        self,
        encoder: torch.nn.Module,
        sa_encoder: torch.nn.Module,
        actor: torch.nn.Module,
        critic: torch.nn.Module,  # TwinCritic 实例，forward 返回 (q1, q2)
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算策略损失

        参数:
            encoder: 状态编码器 φ_s
            sa_encoder: 状态‑动作编码器 φ_sa
            actor: 策略网络，forward 应返回字典，至少包含 "action" 键；
                   若需正则化，还应包含 "pre_activation"（即激活函数前的值）
            critic: 双 Q 网络，forward(z_sa) 返回 (q1, q2)
            batch: 包含以下键的字典
                "obs": (B, *obs_shape) 当前观测

        返回:
            dict 包含:
                "loss":       总损失（标量张量）
                "q1":         Q1 的均值（用于监控）
                "q2":         Q2 的均值
                "pre_act_norm": 预激活值的均方（正则化项，用于监控）
        """
        # 获取当前状态并编码
        obs = batch["obs"]
        z_s = encoder(obs).detach()  # (B, latent_dim)，detach 以避免梯度流回编码器

        # 从策略网络获取动作和预激活值（若存在）
        actor_out = actor(z_s)
        action = actor_out["mean"]           # (B, action_dim)
        pre_activation = actor_out.get("pre_activation", None)

        # 构造状态‑动作嵌入
        z_sa = sa_encoder(z_s, action)         # (B, latent_dim)

        # 计算两个 Q 值
        q1, q2 = critic(z_sa)                  # 各为 (B, 1)

        # 打印取均值用于监控
        print(f"Policy Loss - Q1 mean: {q1.mean().item():.4f}, Q2 mean: {q2.mean().item():.4f}", flush=True)
        # print("Q diff:", (q1 - q2).abs().mean().item())

        # 策略损失：最大化 Q（即最小化负 Q）
        loss_q = -0.5 * (q1 + q2).mean()      # 标量

        # 预激活正则化（若启用且提供了预激活值）
        reg_loss = 0.0
        if pre_activation is not None and self.lambda_pre > 0:
            reg_loss = self.lambda_pre * (pre_activation.pow(2).mean())

        total_loss = loss_q + reg_loss

        # 在计算 loss 后，打印 action 的梯度
        # action.retain_grad()
        # total_loss.backward()
        # print("Action grad mean:", action.grad.abs().mean().item())

        return {
            "loss": total_loss,
            "q1": q1.mean().item(),
            "q2": q2.mean().item(),
            "pre_act_norm": (pre_activation.pow(2).mean().item()
                             if pre_activation is not None else 0.0),
            "action": action,
        }