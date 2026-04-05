import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional

from agents.actor import Actor
from agents.critic import TwinCritic
from models.dynamics import LatentDynamics
from models.encoder import StateEncoder
from models.sa_encoder import StateActionEncoder
from models.heads import TwoHotRewardHead
from losses.representation_loss import RepresentationLoss
from losses.value_loss import ValueLoss
from losses.policy_loss import PolicyLoss
from replay.prioritized_buffer import PrioritizedReplayBuffer
from agents.target_network import TargetNetwork


class ULDAgent:
    """ULD 智能体，整合表示学习、价值函数与策略更新"""

    def __init__(self, cfg: Dict[str, Any]):
        """
        参数:
            cfg: 包含所有超参数的配置字典，例如：
                {
                    "obs_shape": (3, 64, 64),
                    "action_dim": 2,
                    "latent_dim": 64,
                    "encoder_type": "cnn",
                    "hidden_dims": [256, 256],
                    "num_bins": 101,
                    "vmin": -1.0,
                    "vmax": 1.0,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "policy_delay": 2,
                    "H_enc": 5,
                    "H_Q": 5,
                    "lambda_r": 1.0,
                    "lambda_d": 1.0,
                    "lambda_t": 1.0,
                    "lambda_pre": 0.01,
                    "lr_repr": 1e-4,
                    "lr_critic": 3e-4,
                    "lr_actor": 3e-4,
                    "buffer_capacity": 1_000_000,
                    "nstep": 5,
                    "alpha_prior": 0.6,
                    "beta_prior": 0.4,
                    "device": "cuda"
                }
        """
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.obs_shape = cfg["obs_shape"]
        self.action_dim = cfg["action_dim"]
        self.latent_dim = cfg["latent_dim"]
        self.gamma = cfg["gamma"]
        self.tau = cfg["tau"]
        self.policy_delay = cfg["policy_delay"]
        self.H_enc = cfg["H_enc"]
        self.H_Q = cfg["H_Q"]
        self.nstep = max(self.H_enc, self.H_Q)  # 缓冲区存储的最长序列

        # -------------------- 网络模块 --------------------
        # 状态编码器（在线）
        self.encoder = StateEncoder(
            obs_shape=self.obs_shape,
            latent_dim=self.latent_dim,
            encoder_type=cfg["encoder_type"],
            hidden_dims=cfg["hidden_dims"],
        ).to(self.device)

        # 状态‑动作编码器（在线）
        self.sa_encoder = StateActionEncoder(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dims=cfg["hidden_dims"],
        ).to(self.device)

        # 潜在动力学模型（线性预测）
        self.dynamics = LatentDynamics(
            latent_dim=self.latent_dim,
            use_linear=True,
        ).to(self.device)

        # Two‑Hot 奖励头（独立于动力学模型，用于奖励分类）
        self.reward_head = TwoHotRewardHead(
            latent_dim=self.latent_dim,
            num_bins=cfg["num_bins"],
            vmin=cfg["vmin"],
            vmax=cfg["vmax"],
        ).to(self.device)

        # 策略网络
        self.actor = Actor(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dims=cfg["hidden_dims"],
            log_std_min=cfg.get("log_std_min", -10),
            log_std_max=cfg.get("log_std_max", 2),
        ).to(self.device)

        # 双 Q 网络
        self.critic = TwinCritic(
            latent_dim=self.latent_dim,
            hidden_dims=cfg["hidden_dims"],
        ).to(self.device)

        # -------------------- 目标网络（慢更新） --------------------
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # 使用 TargetNetwork 管理 actor 和 critic 的目标网络（软更新）
        self.actor_target = TargetNetwork(self.actor, self.tau)
        self.critic_target = TargetNetwork(self.critic, self.tau)

        # -------------------- 优化器 --------------------
        # 表示学习参数：encoder, sa_encoder, dynamics, reward_head
        self.repr_opt = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.sa_encoder.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.reward_head.parameters()),
            lr=cfg["lr_repr"]
        )

        # Critic 优化器
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg["lr_critic"])

        # Actor 优化器
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg["lr_actor"])

        # -------------------- 损失计算模块 --------------------
        self.repr_loss = RepresentationLoss(
            lambda_r=cfg["lambda_r"],
            lambda_d=cfg["lambda_d"],
            lambda_t=cfg["lambda_t"],
            H_enc=self.H_enc,
            num_bins=cfg["num_bins"],
            vmin=cfg["vmin"],
            vmax=cfg["vmax"],
        )
        self.value_loss = ValueLoss(
            gamma=self.gamma,
            H_Q=self.H_Q,
            use_double_q=True,
        )
        self.policy_loss = PolicyLoss(
            lambda_pre=cfg.get("lambda_pre", 0.0),
        )

        # -------------------- 经验回放 --------------------
        self.buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            capacity=cfg["buffer_capacity"],
            nstep=self.nstep,
            alpha=cfg.get("alpha_prior", 0.6),
            beta=cfg.get("beta_prior", 0.4),
        )

        # 奖励归一化相关变量（运行平均绝对奖励）
        self.reward_scale = 1.0
        self.reward_scale_count = 0

        # 训练步数计数器
        self.total_steps = 0

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(self.device)
            if obs_t.dim() == 3 and self.cfg["encoder_type"] == "mlp":
                obs_t = obs_t.flatten()
            z_s = self.encoder(obs_t.unsqueeze(0))
            if eval_mode:
                action = self.actor(z_s)["mean"]
            else:
                action = self.actor(z_s)["action"]
        return action.squeeze(0).cpu().numpy()  # 返回 [-1, 1] 范围内的动作

    def update_target_networks(self):
        """软更新所有目标网络"""
        # 状态编码器
        with torch.no_grad():
            for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                p_target.data.mul_(1 - self.tau)
                p_target.data.add_(self.tau * p_online.data)
        # Actor 和 Critic
        self.actor_target.update()
        self.critic_target.update()

    def update_reward_scale(self, rewards: torch.Tensor):
        """更新运行平均绝对奖励"""
        abs_r = rewards.abs().mean().item()

        self.reward_scale_count += 1
        # # 使用指数移动平均（简单平均亦可）
        # self.reward_scale = self.reward_scale * (
        #             self.reward_scale_count - 1) / self.reward_scale_count + abs_r / self.reward_scale_count

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        # 将 batch 数据转移到设备
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # 提取用于表示学习和价值学习的部分（根据需要）
        T_seq = batch["obs"].size(1) - 1
        assert T_seq >= max(self.H_enc, self.H_Q)

        repr_batch = {
            "obs": batch["obs"][:, :self.H_enc + 1],
            "actions": batch["actions"][:, :self.H_enc],
            "rewards": batch["rewards"][:, :self.H_enc],
            "dones": batch["dones"][:, :self.H_enc],
        }
        value_batch = {
            "obs": batch["obs"][:, :self.H_Q + 1],
            "actions": batch["actions"][:, :self.H_Q],
            "rewards": batch["rewards"][:, :self.H_Q],
            "dones": batch["dones"][:, :self.H_Q],
        }

        # 更新奖励归一化因子（可选）
        self.update_reward_scale(value_batch["rewards"])

        # =====================================================
        # 1. 表示学习（训练 encoder, sa_encoder, dynamics, reward_head）
        # =====================================================
        repr_loss_dict = self.repr_loss.compute(
            encoder=self.encoder,
            sa_encoder=self.sa_encoder,
            target_encoder=self.target_encoder,
            dynamics=self.dynamics,
            reward_head=self.reward_head,
            batch=repr_batch,
            device=self.device,
            reward_scale=self.reward_scale,
        )
        repr_loss = repr_loss_dict["loss"]

        # =====================================================
        # 2. 价值学习（训练 critic，同时更新 encoder, sa_encoder 的梯度）
        # =====================================================
        value_loss_dict = self.value_loss.compute(
            encoder=self.encoder,
            sa_encoder=self.sa_encoder,
            critic=self.critic,
            critic_target=self.critic_target.target,
            actor_target=self.actor_target.target,
            batch=value_batch,
            reward_scale=self.reward_scale,
        )
        value_loss = value_loss_dict["loss"]

        # 联合损失（可调整权重）
        total_loss = repr_loss + 0.5 * value_loss  # 或 1.0

        self.repr_opt.zero_grad()
        self.critic_opt.zero_grad()
        total_loss.backward()

        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.sa_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.reward_head.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)

        # 更新参数
        self.repr_opt.step()
        self.critic_opt.step()

        # =====================================================
        # 3. 策略学习（只训练 actor，不更新 encoder 和 critic）
        # =====================================================
        if step % self.policy_delay == 0:
            # 1. 冻结 critic（重要！）
            for p in self.critic.parameters():
                p.requires_grad = False

            # 2. 计算策略损失（内部已对 encoder 做 detach）
            policy_loss_dict = self.policy_loss.compute(
                encoder=self.encoder,
                sa_encoder=self.sa_encoder,
                actor=self.actor,
                critic=self.critic,
                batch={"obs": batch["obs"][:, 0]},
            )
            policy_loss = policy_loss_dict["loss"]

            # 3. 更新 actor
            self.actor_opt.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            # 4. 解冻 critic
            for p in self.critic.parameters():
                p.requires_grad = True

        # 软更新目标网络
        self.update_target_networks()

        # 更新优先级（如果有）
        if "indices" in batch and hasattr(self.buffer, "update_priority"):
            td_error = value_loss_dict["td_error"]
            self.buffer.update_priority(batch["indices"], td_error.squeeze(-1).cpu().numpy())

        # 返回监控指标
        return {
            "repr_loss": repr_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item() if step % self.policy_delay == 0 else 0.0,
            "reward_scale": self.reward_scale,
        }

    def add_transition(self, obs, action, reward, next_obs, done):
        """向经验回放添加一条过渡"""
        self.buffer.add(obs, action, reward, next_obs, done)

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """从回放缓冲区采样一个 batch"""
        return self.buffer.sample(batch_size)
