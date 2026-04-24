import torch
import numpy as np
from .replay_buffer import ReplayBuffer   # 假设父类在同一模块

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        obs_shape,
        action_dim,
        capacity: int,
        nstep: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        super().__init__(obs_shape, action_dim, capacity, nstep)
        self.alpha = alpha          # 优先级指数
        self.beta = beta            # 重要性采样指数
        self.priorities = np.ones(capacity, dtype=np.float32)   # 每个物理槽位的优先级
        self.max_priority = 1.0     # 用于初始化新样本的优先级
        self.epsilon = 1e-6         # 避免零优先级

    def add(self, obs, action, reward, next_obs, done):
        # 计算新数据要写入的物理索引（调用父类存储前）
        idx = (self.head + self.size) % self.capacity
        super().add(obs, action, reward, next_obs, done)
        # 设置该槽位的优先级为当前最大优先级（保证新样本有机会被采样）
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int):
        # 有效起始逻辑索引范围 [0, self.size - self.nstep]
        max_start = self.size - self.nstep
        if max_start < 0:
            raise RuntimeError(f"Not enough transitions (need {self.nstep}, have {self.size})")

        logical_indices = np.arange(max_start + 1)          # 所有可能的起始逻辑索引
        # 映射到物理索引（循环缓冲区）
        physical_indices = [(self.head + i) % self.capacity for i in logical_indices]
        # 计算采样概率
        probs = self.priorities[physical_indices] ** self.alpha
        probs /= probs.sum()

        # 按概率采样 batch_size 个逻辑索引
        chosen_logical = np.random.choice(logical_indices, size=batch_size, p=probs)
        chosen_physical = [(self.head + i) % self.capacity for i in chosen_logical]

        # 收集多步序列
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []

        for start_logical in chosen_logical:
            obs_seq = []
            action_seq = []
            reward_seq = []
            done_seq = []
            for t in range(self.nstep):
                logical_idx = start_logical + t
                phys_idx = (self.head + logical_idx) % self.capacity
                obs, action, reward, next_obs, done = self.buffer[phys_idx]
                if t == 0:
                    obs_seq.append(obs)
                obs_seq.append(next_obs)
                action_seq.append(action)
                reward_seq.append(reward)
                done_seq.append(done)
            batch_obs.append(np.stack(obs_seq))
            batch_actions.append(np.stack(action_seq))
            batch_rewards.append(np.array(reward_seq).reshape(-1, 1))
            batch_dones.append(np.array(done_seq).reshape(-1, 1))

        # 转换为张量
        obs_tensor = torch.from_numpy(np.stack(batch_obs)).float()
        actions_tensor = torch.from_numpy(np.stack(batch_actions)).float()
        rewards_tensor = torch.from_numpy(np.stack(batch_rewards)).float()
        dones_tensor = torch.from_numpy(np.stack(batch_dones)).float()

        # 计算重要性采样权重
        N = max_start + 1                       # 有效起始状态总数
        p_chosen = probs[chosen_logical]        # 每个采样起始状态的原始概率
        weights = (1.0 / (N * p_chosen)) ** self.beta
        weights /= weights.max()                # 归一化，使最大权重为1
        weights_tensor = torch.from_numpy(weights).float().view(-1, 1)

        return {
            "obs": obs_tensor,
            "actions": actions_tensor,
            "rewards": rewards_tensor,
            "dones": dones_tensor,
            "indices": chosen_physical,        # 物理索引，用于更新优先级
            "weights": weights_tensor,
        }

    def update_priority(self, indices, td_error):
        """
        更新指定槽位的优先级。
        indices: list of physical indices (与 sample 返回的 indices 对应)
        td_error: torch.Tensor or np.ndarray, shape (batch_size,)
        """
        new_priorities = np.abs(td_error) + self.epsilon
        new_priorities = new_priorities ** self.alpha
        for idx, p in zip(indices, new_priorities):
            self.priorities[idx] = p
        self.max_priority = max(self.max_priority, np.max(new_priorities))