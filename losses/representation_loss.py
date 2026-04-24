import torch
import torch.nn.functional as F
from typing import Dict, Any


class RepresentationLoss:
    """
    表示学习损失类
    实现论文中的多步潜在展开损失，包括：
    - 动力学损失（下一状态嵌入预测）
    - 奖励损失（Two‑Hot 分类）
    - 终止损失（done 标志预测）
    """

    def __init__(
        self,
        lambda_r: float,          # 奖励损失系数
        lambda_d: float,          # 动力学损失系数
        lambda_t: float,          # 终止损失系数
        H_enc: int,               # 展开步数
        num_bins: int,            # Two‑Hot 编码的 bin 数量
        vmin: float,              # 奖励最小值（原始尺度）
        vmax: float,              # 奖励最大值（原始尺度）
    ):
        self.lambda_r = lambda_r
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.H_enc = H_enc
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        """
        symexp 变换：sign(x) * (exp(|x|) - 1)
        将原始奖励映射到变换空间，实现非均匀 bin 划分
        """
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def _two_hot_target(self, rewards, bin_centers_trans, spacing):
        rewards_trans = self.symexp(rewards)
        rewards_trans = torch.clamp(rewards_trans, bin_centers_trans[0], bin_centers_trans[-1])
        pos = (rewards_trans - bin_centers_trans[0]) / spacing
        lower_idx = torch.floor(pos).long()
        upper_idx = torch.ceil(pos).long()
        lower_idx = torch.clamp(lower_idx, 0, self.num_bins - 1)
        upper_idx = torch.clamp(upper_idx, 0, self.num_bins - 1)

        upper_weight = pos - lower_idx.float()
        lower_weight = 1.0 - upper_weight

        two_hot = torch.zeros(rewards.size(0), self.num_bins, device=rewards.device)
        same = (lower_idx == upper_idx)
        diff_mask = ~same

        # 不同索引：分配两个权重
        two_hot[diff_mask, lower_idx[diff_mask]] = lower_weight[diff_mask]
        two_hot[diff_mask, upper_idx[diff_mask]] = upper_weight[diff_mask]

        # 相同索引：只分配一个权重 1
        two_hot[same, lower_idx[same]] = 1.0

        return two_hot

    def compute(
        self,
        encoder: torch.nn.Module,           # 状态编码器 φ_s (当前)
        sa_encoder: torch.nn.Module,        # 状态‑动作编码器 φ_sa
        target_encoder: torch.nn.Module,    # 目标状态编码器（慢更新，提供稳定的目标）
        dynamics: torch.nn.Module,          # 线性环境模型，预测 z_next、reward_logits、done
        reward_head: torch.nn.Module,       # Two‑Hot 奖励头（若 dynamics 已输出 reward_logits 则可设为 None）
        batch: Dict[str, torch.Tensor],     # 采样的过渡序列
        device: torch.device,               # 计算设备
        reward_scale: float = 1.0,  # 新增参数，表示运行平均绝对奖励
    ) -> Dict[str, torch.Tensor]:
        """
        通过多步潜在展开计算表示学习损失

        参数:
            encoder: 当前状态编码器 φ_s
            sa_encoder: 当前状态‑动作编码器 φ_sa
            target_encoder: 目标状态编码器（用于稳定目标）
            dynamics: 线性环境模型，输入 z_sa，输出字典 {"z_next", "reward_logits", "done"}
            reward_head: Two‑Hot 奖励头（可选，若 dynamics 已包含奖励头则忽略）
            batch: 包含以下键的字典
                "obs":    (B, H_enc+1, ...) 观测序列
                "actions": (B, H_enc, A)    动作序列
                "rewards": (B, H_enc, 1)    奖励序列（原始标量）
                "dones":   (B, H_enc, 1)    终止标志序列
            device: torch 设备

        返回:
            dict 包含:
                "loss":       总损失（加权和）
                "loss_dyn":   动力学损失（MSE）
                "loss_reward":奖励损失（交叉熵）
                "loss_done":  终止损失（MSE）
        """
        B, T, *obs_shape = batch["obs"].shape
        assert T == self.H_enc + 1, f"观测序列长度应为 {self.H_enc+1}，实际为 {T}"
        actions = batch["actions"]          # (B, H_enc, A)
        rewards = batch["rewards"].squeeze(-1)  # (B, H_enc)
        dones = batch["dones"].squeeze(-1)      # (B, H_enc)

        # 预计算变换空间中的 bin 中心和间距（用于生成 Two‑Hot 目标）
        with torch.no_grad():
            vmin_t = self.symexp(torch.tensor(self.vmin, device=device))
            vmax_t = self.symexp(torch.tensor(self.vmax, device=device))
            bin_centers_trans = torch.linspace(vmin_t, vmax_t, self.num_bins, device=device)
            spacing = (vmax_t - vmin_t) / (self.num_bins - 1)

        # 初始状态编码
        z_s = encoder(batch["obs"][:, 0])          # (B, latent_dim)

        loss_dyn = 0.0
        loss_reward = 0.0
        loss_done = 0.0

        # 多步展开（H_enc 步）
        for t in range(self.H_enc):
            # ----- 1. 构造状态‑动作嵌入 -----
            z_sa = sa_encoder(z_s, actions[:, t])   # (B, latent_dim)

            # ----- 2. 通过动力学模型预测 -----
            pred = dynamics(z_sa)                   # 返回字典
            z_next_pred = pred["z_next"]            # (B, latent_dim)
            reward_logits_pred = pred.get("reward_logits")
            done_pred = pred["done"].squeeze(-1)     # (B,)

            # 若 dynamics 未输出奖励 logits，则使用独立的 reward_head
            if reward_logits_pred is None and reward_head is not None:
                reward_logits_pred = reward_head(z_sa)  # (B, num_bins)

            # ----- 3. 获取目标值（使用目标编码器，无梯度）-----
            with torch.no_grad():
                z_true_next = target_encoder(batch["obs"][:, t+1])   # (B, latent_dim)
                # 将原始奖励归一化
                norm_reward = rewards[:, t] / reward_scale
                # 将真实奖励转换为 Two‑Hot 软标签
                two_hot_target = self._two_hot_target(norm_reward, bin_centers_trans, spacing)

            # 打印two_hot_target
            # print(f"two_hot_target (step {t}): {two_hot_target[0].cpu().numpy()}")

            # ----- 4. 计算各项损失 -----
            # 动力学损失：预测的下一状态嵌入与真实嵌入之间的 MSE
            loss_dyn += F.mse_loss(z_next_pred, z_true_next)

            # 奖励损失：预测分布与 Two‑Hot 目标之间的交叉熵
            if reward_logits_pred is not None:
                log_probs = F.log_softmax(reward_logits_pred, dim=-1)
                loss_reward += -(two_hot_target * log_probs).sum(dim=-1).mean()

            # 终止损失：预测的 done 与真实 done 之间的 MSE
            loss_done += F.mse_loss(done_pred, dones[:, t])

            # ----- 5. 更新当前状态为预测的下一状态（自回归）-----
            # 使用 detach() 阻止梯度回传，避免训练不稳定（论文推荐）
            z_s = z_next_pred.detach()

        # 加权组合三项损失
        total_loss = (self.lambda_r * loss_reward +
                      self.lambda_d * loss_dyn +
                      self.lambda_t * loss_done)

        return {
            "loss": total_loss,
            "loss_dyn": loss_dyn,
            "loss_reward": loss_reward,
            "loss_done": loss_done,
        }