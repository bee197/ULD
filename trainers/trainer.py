import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np
import torch
from typing import Dict, Any, List
from agents.uld_agent import ULDAgent
from replay.prioritized_buffer import PrioritizedReplayBuffer


class Trainer:
    """
    ULD 智能体训练器，支持多环境并行采集数据
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # 并行环境数量
        self.num_envs = cfg.get("num_envs", 4)

        # 创建异步并行环境（多进程）
        def make_env(seed_offset=0):
            def _thunk():
                env = gym.make(cfg["env_name"])
                # 注意：AsyncVectorEnv 无法直接传递种子，需要在每个子进程中通过环境变量设置
                return env

            return _thunk

        self.env = AsyncVectorEnv([make_env() for _ in range(self.num_envs)])

        self.single_obs_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.obs_shape = self.single_obs_space.shape
        self.action_dim = self.single_action_space.shape[0]

        # 评估环境（单环境）
        self.eval_env = gym.make(cfg["env_name"])

        # 更新配置中的观测维度和动作维度
        cfg["obs_shape"] = self.obs_shape
        cfg["action_dim"] = self.action_dim
        cfg["device"] = self.device

        # 创建 ULD 智能体（内部包含回放缓冲区）
        self.agent = ULDAgent(cfg)

        # 训练计数器（总步数为所有环境步数之和）
        self.total_steps = 0
        self.start_steps = cfg["start_steps"]
        self.update_every = cfg.get("update_every", 1)
        self.eval_interval = cfg.get("eval_interval", 5000)
        self.eval_episodes = cfg.get("eval_episodes", 10)
        self.render_eval = cfg.get("render_eval", False)

        # 每个环境的当前状态
        self.obs, _ = self.env.reset(seed=cfg.get("seed"))
        # 每个环境的 episode 累计奖励和步数
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_counts = np.zeros(self.num_envs, dtype=np.int32)
        self.total_episodes = 0

        # 随机种子
        if "seed" in cfg:
            np.random.seed(cfg["seed"])
            torch.manual_seed(cfg["seed"])
            # 注意：SyncVectorEnv 的种子需要在 reset 时传递，已在上面传入

        # 在 Trainer.__init__ 中，创建多个 buffer
        self.buffers = [
            PrioritizedReplayBuffer(
                obs_shape=self.obs_shape,
                action_dim=self.action_dim,
                capacity=self.cfg["buffer_capacity"] // self.num_envs,
                nstep=self.cfg["nstep"],
                alpha=self.cfg.get("alpha_prior", 0.0),
                beta=self.cfg.get("beta_prior", 1.0),
            ) for _ in range(self.num_envs)
        ]

    def collect_step(self) -> Dict[str, float]:
        """
        并行收集所有环境一步，将 transitions 存入回放缓冲区。
        返回所有环境的平均奖励等信息。
        """
        # 1. 为每个环境选择归一化动作 [-1, 1]
        norm_actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        if self.total_steps < self.start_steps:
            # 随机探索阶段：从每个环境的动作空间采样，然后归一化
            for i in range(self.num_envs):
                env_action = self.single_action_space.sample()
                norm_action = env_action / self.single_action_space.high
                norm_actions[i] = norm_action
        else:
            # 使用策略网络
            for i in range(self.num_envs):
                obs_i = self.obs[i]
                norm_action = self.agent.act(obs_i, eval_mode=False)
                norm_actions[i] = norm_action

        # 2. 缩放动作到环境实际范围
        norm_actions = np.clip(norm_actions, -1.0, 1.0)
        env_actions = norm_actions * self.single_action_space.high

        # 3. 异步执行 step
        self.env.step_async(env_actions)
        next_obs, rewards, terminations, truncations, infos = self.env.step_wait()
        dones = terminations | truncations

        # 4. 存储每个环境的 transition 并更新统计
        for i in range(self.num_envs):
            # 存储时使用对应环境 buffer
            self.buffers[i].add(
                self.obs[i],
                norm_actions[i],
                rewards[i],
                next_obs[i],
                dones[i],
            )

            # 更新 episode 统计
            self.episode_rewards[i] += rewards[i]
            self.episode_lengths[i] += 1

            if dones[i]:
                # 打印该环境的 episode 结果
                print(f"Env {i} Episode {self.episode_counts[i]} finished after {self.episode_lengths[i]} steps, reward: {self.episode_rewards[i]:.2f}")
                self.episode_counts[i] += 1
                self.total_episodes += 1
                # 重置该环境的统计（环境已在 step 后自动重置？SyncVectorEnv 会自动重置 done 的环境）
                # 但我们需要重置累计奖励和步数计数器
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0

        # 更新当前观测为 next_obs
        self.obs = next_obs
        # 总步数增加 num_envs
        self.total_steps += self.num_envs

        return {
            "reward": rewards.mean(),
            "done": dones.any(),
            "episode_reward_mean": self.episode_rewards.mean(),
            "episode_length_mean": self.episode_lengths.mean(),
        }

    def train_step(self):
        if self.total_steps < self.start_steps:
            return {}
        env_id = np.random.randint(self.num_envs)
        if self.buffers[env_id].size < self.buffers[env_id].nstep + 1:
            return {}
        batch = self.buffers[env_id].sample(self.cfg["batch_size"])
        return self.agent.update(batch, self.total_steps)

    def evaluate(self) -> float:
        """评估当前策略（单环境）"""
        total_reward = 0.0
        for ep in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                if self.render_eval:
                    self.eval_env.render()
                norm_action = self.agent.act(obs, eval_mode=True)
                norm_action = np.clip(norm_action, -1.0, 1.0)
                env_action = norm_action * self.eval_env.action_space.high
                obs, reward, terminated, truncated, _ = self.eval_env.step(env_action)
                done = terminated or truncated
                episode_reward += reward
            total_reward += episode_reward
        mean_reward = total_reward / self.eval_episodes
        print(f"Evaluation after {self.total_steps} steps: mean reward = {mean_reward:.2f}")
        with open("uld_eval_log.csv", "a") as f:
            f.write(f"{self.total_steps},{mean_reward}\n")
        return mean_reward

    def train(self):
        """主训练循环"""
        print("Starting training...")
        # 写入 CSV 表头（如果文件不存在或为空）
        try:
            with open("uld_eval_log.csv", "x") as f:
                f.write("step,mean_reward\n")
        except FileExistsError:
            pass

        self.evaluate()  # 初始评估

        while self.total_steps < self.cfg["total_steps"]:
            collect_info = self.collect_step()

            # 定期评估
            if self.total_steps % self.eval_interval == 0:
                self.evaluate()

            # 训练更新
            if self.total_steps >= self.start_steps and self.total_steps % self.update_every == 0:
                train_metrics = self.train_step()
                if train_metrics and self.total_steps % 1000 == 0:
                    print(f"Step {self.total_steps}, train metrics: {train_metrics}")


        print("Training finished.")
        self.evaluate()
        self.env.close()
        self.eval_env.close()