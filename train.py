import gym
from model import ULD, ReplayBuffer

env = gym.make("HalfCheetah-v4")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = ULD(state_dim, action_dim)
buffer = ReplayBuffer(state_dim, action_dim)

s = env.reset()

for t in range(1_000_000):

    a = agent.act(s)

    s2, r, done, _ = env.step(a)

    buffer.add(s, a, s2, r, done)

    s = s2

    if done:
        s = env.reset()

    if buffer.size > 10000:
        agent.train(buffer)