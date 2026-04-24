import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, Optional
from agents.uld_agent import ULDAgent


class Trainer:
    """
    ULD 智能体训练器，负责环境交互、数据收集和训练循环
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        参数:
            cfg: 配置字典，应包含以下键：
                - env_name: 环境名称
                - seed: 随机种子
                - total_steps: 总训练步数（环境交互次数）
                - start_steps: 开始训练前的随机探索步数
                - update_every: 每收集多少步后进行一次训练更新
                - eval_interval: 每多少步进行一次评估
                - eval_episodes: 评估时的回合数
                - render_eval: 是否在评估时渲染
                - 以及其他 ULDAgent 需要的配置
        """
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # 创建环境
        self.env = gym.make(cfg["env_name"])
        self.eval_env = gym.make(cfg["env_name"]) if cfg.get("eval_env_name") is None else gym.make(cfg["eval_env_name"])
        self.obs_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0] if hasattr(self.env.action_space, 'shape') else self.env.action_space.n

        # 更新配置中的观测维度和动作维度
        cfg["obs_shape"] = self.obs_shape
        cfg["action_dim"] = self.action_dim
        cfg["device"] = self.device

        # 创建 ULD 智能体（内部包含回放缓冲区）
        self.agent = ULDAgent(cfg)

        # 训练计数器
        self.total_steps = 0                 # 总环境交互步数
        self.start_steps = cfg["start_steps"]
        self.update_every = cfg.get("update_every", 1)
        self.eval_interval = cfg.get("eval_interval", 5000)
        self.eval_episodes = cfg.get("eval_episodes", 10)
        self.render_eval = cfg.get("render_eval", False)

        # 记录当前回合状态
        self.obs, _ = self.env.reset(seed=cfg.get("seed"))
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_count = 0

        # 随机种子
        if "seed" in cfg:
            np.random.seed(cfg["seed"])
            torch.manual_seed(cfg["seed"])
            self.env.action_space.seed(cfg["seed"])

    def collect_step(self) -> Dict[str, float]:
        """
        执行一个环境步，将 transition 存入回放缓冲区，并返回该步的奖励等信息
        """
        # 1. 获取归一化动作 [-1, 1]
        if self.total_steps < self.start_steps:
            # 随机采样环境动作，再归一化到 [-1,1]
            env_action = self.env.action_space.sample()
            # 假设动作空间对称，归一化公式：norm_action = env_action / high
            norm_action = env_action / self.env.action_space.high
        else:
            norm_action = self.agent.act(self.obs, eval_mode=False)  # 已经是 [-1,1]

        # 2.TODO:缩放动作到环境实际范围
        env_action = norm_action * self.env.action_space.high

        # 执行动作
        next_obs, reward, terminated, truncated, info = self.env.step(env_action)
        # 打印Raw reward
        # print(f"Raw reward from env: {reward}")
        done = terminated or truncated

        # 存储 transition
        self.agent.add_transition(self.obs, norm_action, reward, next_obs, done)

        # 更新累计奖励和长度
        self.episode_reward += reward
        self.episode_length += 1

        # 处理回合结束
        if done:
            # 记录回合统计
            print(f"Episode {self.episode_count} finished after {self.episode_length} steps, reward: {self.episode_reward:.2f}")
            # 重置环境
            self.obs, _ = self.env.reset()
            self.episode_reward = 0.0
            self.episode_length = 0
            self.episode_count += 1
        else:
            self.obs = next_obs

        self.total_steps += 1

        return {
            "reward": reward,
            "done": done,
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
        }

    def train_step(self) -> Dict[str, float]:
        """
        执行一次模型更新（从回放缓冲区采样并更新网络）
        """
        if self.agent.buffer.size < self.agent.buffer.nstep + 1:
            # 缓冲区数据不足，跳过更新
            return {}

        batch = self.agent.sample_batch(self.cfg.get("batch_size", 256))
        metrics = self.agent.update(batch, self.total_steps)
        return metrics

    def evaluate(self) -> float:
        """
        评估当前策略，返回平均回合奖励
        """
        total_reward = 0.0
        for ep in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                if self.render_eval:
                    self.eval_env.render()
                action = self.agent.act(obs, eval_mode=True)
                # TODO:缩放动作到环境实际范围
                action = action * self.eval_env.action_space.high
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            total_reward += episode_reward
        mean_reward = total_reward / self.eval_episodes
        print(f"Evaluation after {self.total_steps} steps: mean reward = {mean_reward:.2f}")
        # 追加写入日志
        with open("uld_eval_log.csv", "a") as f:
            f.write(f"{self.total_steps},{mean_reward}\n")
        return mean_reward

    def train(self):
        """
        主训练循环
        """
        print("Starting training...")
        self.evaluate()  # 初始评估

        while self.total_steps < self.cfg["total_steps"]:
            # 收集一个环境步
            collect_info = self.collect_step()

            # 每 update_every 步执行一次训练更新
            if self.total_steps >= self.start_steps and self.total_steps % self.update_every == 0:
                train_metrics = self.train_step()

                # 可选：打印训练指标
                if self.total_steps % 1000 == 0:
                    print(f"Step {self.total_steps}, train metrics: {train_metrics}")

            # 定期评估
            if self.total_steps % self.eval_interval == 0:
                self.evaluate()

        print("Training finished.")
        self.evaluate()  # 最终评估
        self.env.close()
        self.eval_env.close()