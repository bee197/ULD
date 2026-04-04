import random
import torch
import numpy as np
from collections import deque
from typing import Union, Optional, Dict, List, Any


class ReplayBuffer:
    def __init__(
        self,
        obs_shape: Union[tuple, int],
        action_dim: int,
        capacity: int,
        nstep: int = 1,
    ):
        """
        参数:
            obs_shape: 观测空间形状，用于预先分配内存（可选）
            action_dim: 动作维度
            capacity: 缓冲区最大容量
            nstep: 多步采样长度（返回 nstep 个连续过渡）
        """
        self.capacity = capacity
        self.nstep = nstep
        self.buffer = [None] * capacity
        self.head = 0          # 最旧数据的逻辑索引
        self.size = 0          # 当前存储的有效数据量

        # 用于快速构建张量的辅助属性（可选）
        self.obs_shape = obs_shape
        self.action_dim = action_dim

    def add(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        reward: float,
        next_obs: Union[np.ndarray, torch.Tensor],
        done: bool,
    ) -> None:
        """存储一个时间步的过渡"""
        # 转换为 numpy 数组便于存储（可根据需求保留张量）
        if torch.is_tensor(obs):
            obs = obs.cpu().numpy()
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        if torch.is_tensor(next_obs):
            next_obs = next_obs.cpu().numpy()

        data = (obs, action, reward, next_obs, done)

        # 计算实际存储位置
        idx = (self.head + self.size) % self.capacity
        self.buffer[idx] = data

        if self.size < self.capacity:
            self.size += 1
        else:
            # 缓冲区已满，覆盖最旧数据，移动 head
            self.head = (self.head + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        从缓冲区采样一个 batch，每个样本包含连续的 nstep 个过渡。

        返回字典:
            "obs":      [B, nstep+1, *obs_shape]  观测序列
            "actions":  [B, nstep, action_dim]    动作序列
            "rewards":  [B, nstep, 1]             奖励序列
            "dones":    [B, nstep, 1]             终止标志序列
        """
        if self.size < self.nstep:
            raise RuntimeError(f"Not enough transitions (need {self.nstep}, have {self.size})")

        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []

        # 随机选择起始索引（逻辑索引 0..size-nstep）
        start_indices = random.sample(range(self.size - self.nstep + 1), batch_size)

        for start in start_indices:
            obs_seq = []
            action_seq = []
            reward_seq = []
            done_seq = []

            # 收集 nstep 个过渡
            for i in range(self.nstep):
                # 计算实际存储位置
                real_idx = (self.head + start + i) % self.capacity
                obs, action, reward, next_obs, done = self.buffer[real_idx]

                if i == 0:
                    # 第一个状态是当前 obs
                    obs_seq.append(obs)
                # 每个过渡都贡献一个 next_obs（即下一个状态）
                obs_seq.append(next_obs)
                action_seq.append(action)
                reward_seq.append(reward)
                done_seq.append(done)

            # 转换为张量并添加至 batch
            batch_obs.append(np.stack(obs_seq))          # (nstep+1, *obs_shape)
            batch_actions.append(np.stack(action_seq))   # (nstep, action_dim)
            batch_rewards.append(np.array(reward_seq).reshape(-1, 1))
            batch_dones.append(np.array(done_seq).reshape(-1, 1))

        # 堆叠成 batch
        obs_tensor = torch.from_numpy(np.stack(batch_obs)).float()
        actions_tensor = torch.from_numpy(np.stack(batch_actions)).float()
        rewards_tensor = torch.from_numpy(np.stack(batch_rewards)).float()
        dones_tensor = torch.from_numpy(np.stack(batch_dones)).float()

        return {
            "obs": obs_tensor,      # [B, nstep+1, ...]
            "actions": actions_tensor,  # [B, nstep, action_dim]
            "rewards": rewards_tensor,  # [B, nstep, 1]
            "dones": dones_tensor,      # [B, nstep, 1]
        }