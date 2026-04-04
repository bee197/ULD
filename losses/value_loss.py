import torch
import torch.nn.functional as F
from typing import Dict


class ValueLoss:
    """
    价值函数损失类
    实现 ULD 论文中的多步 TD 目标与 Huber 损失
    """

    def __init__(
        self,
        gamma: float,           # 折扣因子
        H_Q: int,               # 多步回报长度
        use_double_q: bool = True,   # 是否使用双 Q 技巧（取两网络较小值）
        target_policy_noise=0.2,
        target_noise_clip=0.5
    ):
        self.gamma = gamma
        self.H_Q = H_Q
        self.use_double_q = use_double_q
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

    def _compute_multistep_target(
        self,
        rewards: torch.Tensor,          # (B, H_Q)
        dones: torch.Tensor,            # (B, H_Q)
        next_q_target: torch.Tensor,    # (B, 1)
    ) -> torch.Tensor:
        """
        计算多步 TD 目标值（标量）

        参数:
            rewards: 各步奖励（原始值，尚未归一化）
            dones:   各步终止标志（1 表示终止）
            next_q_target: 最后一时刻的 Q 目标值（已经过 reward scaling）

        返回:
            target: (B, 1) 多步回报
        """
        B = rewards.size(0)
        target = next_q_target.clone()   # (B, 1)

        # 从后向前累计折扣奖励（处理 done 时截断）
        discount = self.gamma
        for t in reversed(range(self.H_Q)):
            # 奖励部分
            reward = rewards[:, t].unsqueeze(-1)  # (B, 1)
            # 如果当前步已经终止，则后续不累积
            target = reward + discount * (1 - dones[:, t].unsqueeze(-1)) * target
            discount *= self.gamma

        return target

    def compute(
        self,
        encoder: torch.nn.Module,           # 当前状态编码器 φ_s
        sa_encoder: torch.nn.Module,        # 当前状态‑动作编码器 φ_sa
        critic: torch.nn.Module,            # 当前双 Q 网络（forward 返回 q1, q2）
        critic_target: torch.nn.Module,     # 目标双 Q 网络
        actor_target: torch.nn.Module,      # 目标策略网络（用于选择下一动作）
        batch: Dict[str, torch.Tensor],     # 采样过渡序列
        reward_scale: float = 1.0,          # 运行平均绝对奖励（用于缩放目标 Q）
    ) -> Dict[str, torch.Tensor]:
        """
        计算价值损失

        参数:
            encoder:        当前状态编码器
            sa_encoder:     当前状态‑动作编码器
            critic:         当前 Q 网络（用于当前 Q 值）
            critic_target:  目标 Q 网络（用于目标值）
            actor_target:   目标策略网络（用于下一动作选择）
            batch:          包含以下键的字典
                "obs":     (B, H_Q+1, *obs_shape)  观测序列
                "actions": (B, H_Q, action_dim)    动作序列
                "rewards": (B, H_Q, 1)             奖励序列（原始值）
                "dones":   (B, H_Q, 1)             终止标志序列
            reward_scale:   运行平均绝对奖励（用于缩放目标 Q）

        返回:
            dict 包含:
                "loss":       Huber 损失（标量）
                "q_target":   目标值的均值（用于监控）
                "q_pred":     当前 Q 值的均值（用于监控）
        """
        B, T, *obs_shape = batch["obs"].shape
        assert T == self.H_Q + 1, f"观测序列长度应为 {self.H_Q+1}，实际为 {T}"

        # 提取序列组件
        obs_seq = batch["obs"]                     # (B, H_Q+1, ...)
        actions_seq = batch["actions"]             # (B, H_Q, action_dim)
        rewards_seq = batch["rewards"].squeeze(-1) # (B, H_Q)
        dones_seq = batch["dones"].squeeze(-1)     # (B, H_Q)

        # --------------------------------
        # 1. 计算当前 Q 值（用于损失）
        # --------------------------------
        # 当前状态编码
        z_s = encoder(obs_seq[:, 0])               # (B, latent_dim)
        # 打印z_s
        # print(f"z_s: min={z_s.min():.2f}, max={z_s.max():.2f}, std={z_s.std():.2f}")
        # 当前动作
        current_actions = actions_seq[:, 0]        # (B, action_dim)
        # 状态‑动作嵌入
        # z_sa = sa_encoder(z_s, current_actions)    # (B, latent_dim)
        # 当前 Q 值（两个网络输出）
        q1_pred, q2_pred = critic(z_s, current_actions)            # 各为 (B, 1)
        # 打印q
        # print(f"q1_pred: {q1_pred.mean().item():.4f}, q2_pred: {q2_pred.mean().item():.4f}")

        # --------------------------------
        # 2. 计算多步 TD 目标
        # --------------------------------
        with torch.no_grad():
            # 2.1 编码最后一时刻状态（用于计算下一 Q 目标）
            z_next = encoder(obs_seq[:, -1])       # (B, latent_dim)

            # 2.2 通过目标策略选择下一动作
            actor_out = actor_target(z_next)
            next_action = actor_out["action"]      # (B, action_dim)

            # -------------------- 添加目标策略平滑噪声 --------------------
            if self.target_policy_noise > 0:
                noise = torch.randn_like(next_action) * self.target_policy_noise
                noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
                next_action = (next_action + noise).clamp(-1, 1)

            # 2.4 目标 Q 值（双 Q 取最小值）
            q1_next, q2_next = critic_target(z_next, next_action)  # (B, 1)
            if self.use_double_q:
                q_next = torch.min(q1_next, q2_next)     # (B, 1)
            else:
                q_next = q1_next

            # 2.5 缩放目标 Q（论文中 ˜r * Q_target）
            q_next_scaled = q_next / reward_scale

            # 2.6 计算多步回报（考虑 done 截断）
            norm_rewards_seq = rewards_seq / reward_scale
            target = self._compute_multistep_target(
                norm_rewards_seq,            # 归一化奖励 (B, H_Q)
                dones_seq,              # (B, H_Q)
                q_next_scaled,          # (B, 1)
            )  # (B, 1)

        # --------------------------------
        # 3. 损失计算（Huber）
        # --------------------------------
        # 当前 Q 值（取两网络的最小值用于损失，但实际两个网络都训练，损失分别计算）
        # 论文中使用两个 Q 网络，分别与同一个目标比较
        loss_q1 = F.huber_loss(q1_pred, target)
        loss_q2 = F.huber_loss(q2_pred, target)
        loss_total = loss_q1 + loss_q2

        with torch.no_grad():
            # 使用 q1 的误差（或 q2，两者相近）
            td_error = (target - q1_pred).abs().detach()

        return {
            "loss": loss_total,
            "q_target": target.mean().item(),
            "q_pred": (q1_pred.mean().item() + q2_pred.mean().item()) / 2.0,
            "td_error": td_error,  # (B, 1)
        }