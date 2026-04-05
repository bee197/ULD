import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # 解决 PyCharm 后端错误
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import pandas as pd
import os

# ==================== 配置 ====================
ENV_NAME = "Pendulum-v1"
TOTAL_TIMESTEPS = 1_000_000  # 与你的 ULD 训练总步数一致
EVAL_FREQ = 5_000  # 评估间隔（环境步数）
N_EVAL_EPISODES = 10  # 评估回合数
SEED = 42

# ULD 评估日志路径（假设你已经保存了每 eval_interval 步的平均奖励）
# 文件格式：step, mean_reward
ULD_LOG_PATH = "uld_eval_log.csv"  # 你需要根据实际路径修改

# ==================== 训练 PPO ====================
# 创建向量环境（并行4个，加快采样）
env = make_vec_env(ENV_NAME, n_envs=4, seed=SEED)

# 创建回调：定期评估并记录结果
eval_env = make_vec_env(ENV_NAME, n_envs=1, seed=SEED)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./ppo_best/",
    log_path="./ppo_logs/",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=0
)

# 初始化 PPO 模型（超参数可调整）
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=SEED,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/"
)

# print("开始训练 PPO...")
# model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
# model.save("ppo_pendulum")

# 从回调日志中读取评估结果
# SB3 的 EvalCallback 会在 log_path 下生成一个 .monitor.csv 文件
# 我们直接读取该文件
eval_log_file = f"./ppo_logs/evaluations.npz"
if os.path.exists(eval_log_file):
    data = np.load(eval_log_file)
    steps = data['timesteps']
    rewards = data['results'].mean(axis=1)  # 每个评估点平均奖励
else:
    # 如果没有生成，手动模拟（一般会自动生成）
    print("未找到评估日志，请检查 log_path 设置。")
    steps, rewards = [], []

# ==================== 读取 ULD 评估数据 ====================
if os.path.exists(ULD_LOG_PATH):
    uld_df = pd.read_csv(ULD_LOG_PATH)
    uld_steps = uld_df['step'].values
    uld_rewards = uld_df['mean_reward'].values
else:
    print(f"ULD 日志文件 {ULD_LOG_PATH} 不存在，将使用模拟数据演示。")
    # 模拟数据（仅供参考）
    uld_steps = np.arange(0, TOTAL_TIMESTEPS + 1, EVAL_FREQ)
    # 模拟 ULD 从 -1600 逐渐上升到 -200
    uld_rewards = -1600 + 1400 * (uld_steps / TOTAL_TIMESTEPS) + np.random.normal(0, 50, len(uld_steps))
    uld_rewards = np.clip(uld_rewards, -1600, -200)


def moving_average(data, window=5):
    """计算移动平均，保持长度不变（前 window-1 个点用原始值）"""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    # 保持长度一致，前面补 NaN 或原始值，这里补原始值
    return np.concatenate((data[:window - 1], smoothed))


# 平滑窗口大小，可根据需要调整
SMOOTH_WINDOW = 5

# 对 PPO 数据平滑
if steps.size > 0:
    smooth_rewards = moving_average(rewards, SMOOTH_WINDOW)
    plt.plot(steps, smooth_rewards, label='PPO (SB3)', color='blue', linewidth=2)

# 对 ULD 数据平滑
smooth_uld_rewards = moving_average(uld_rewards, SMOOTH_WINDOW)
plt.plot(uld_steps, smooth_uld_rewards, label='ULD (Ours)', color='red', linestyle='--', linewidth=2)

# ==================== 绘制对比图 ====================
plt.figure(figsize=(10, 6))
if steps.size > 0:
    plt.plot(steps, rewards, label='PPO (SB3)', color='blue', linewidth=2)
plt.plot(uld_steps, uld_rewards, label='ULD (Ours)', color='red', linestyle='--', linewidth=2)
plt.xlabel('Environment Steps')
plt.ylabel('Mean Episode Reward (Original)')
plt.title(f'Performance Comparison on {ENV_NAME}')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.xlim(0, 300000)  # 横坐标 0 到 40k
plt.savefig('comparison_plot.png', dpi=150)
plt.show()
print("对比图已保存为 comparison_plot.png")
