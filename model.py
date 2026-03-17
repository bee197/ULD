import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Replay Buffer
# ==========================================

class ReplayBuffer:

    def __init__(self, state_dim, action_dim, size=int(1e6)):

        self.s = np.zeros((size, state_dim))
        self.a = np.zeros((size, action_dim))
        self.s2 = np.zeros((size, state_dim))
        self.r = np.zeros((size, 1))
        self.d = np.zeros((size, 1))

        self.ptr = 0
        self.size = 0
        self.max_size = size

    def add(self, s, a, s2, r, d):

        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.s2[self.ptr] = s2
        self.r[self.ptr] = r
        self.d[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch):

        idx = np.random.randint(0, self.size, size=batch)

        return (
            torch.FloatTensor(self.s[idx]).to(device),
            torch.FloatTensor(self.a[idx]).to(device),
            torch.FloatTensor(self.s2[idx]).to(device),
            torch.FloatTensor(self.r[idx]).to(device),
            torch.FloatTensor(self.d[idx]).to(device)
        )


# ==========================================
# Encoder φ_s
# ==========================================

class StateEncoder(nn.Module):

    def __init__(self, state_dim, latent_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, s):

        return self.net(s)


# ==========================================
# State-Action Encoder φ_sa
# ==========================================

class StateActionEncoder(nn.Module):

    def __init__(self, latent_dim, action_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, z_s, a):

        x = torch.cat([z_s, a], dim=-1)
        return self.net(x)


# ==========================================
# Linear Latent Environment Model
# ==========================================

class LatentDynamics(nn.Module):

    def __init__(self, latent_dim):

        super().__init__()

        self.next_z = nn.Linear(latent_dim, latent_dim)
        self.reward = nn.Linear(latent_dim, 1)
        self.done = nn.Linear(latent_dim, 1)

    def forward(self, z_sa):

        z_next = self.next_z(z_sa)
        r = self.reward(z_sa)
        d = torch.sigmoid(self.done(z_sa))

        return z_next, r, d


# ==========================================
# Critic Q(z_sa)
# ==========================================

class Critic(nn.Module):

    def __init__(self, latent_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, z_sa):

        return self.net(z_sa)


# ==========================================
# Actor π(z_s)
# ==========================================

class Actor(nn.Module):

    def __init__(self, latent_dim, action_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, z_s):

        return self.net(z_s)


# ==========================================
# ULD Agent
# ==========================================

class ULD:

    def __init__(self, state_dim, action_dim):

        latent_dim = 128

        self.encoder = StateEncoder(state_dim, latent_dim).to(device)
        self.sa_encoder = StateActionEncoder(latent_dim, action_dim).to(device)

        self.dynamics = LatentDynamics(latent_dim).to(device)

        self.actor = Actor(latent_dim, action_dim).to(device)
        self.actor_target = Actor(latent_dim, action_dim).to(device)

        self.critic1 = Critic(latent_dim).to(device)
        self.critic2 = Critic(latent_dim).to(device)

        self.critic1_target = Critic(latent_dim).to(device)
        self.critic2_target = Critic(latent_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.sa_encoder.parameters()) +
            list(self.dynamics.parameters()),
            lr=3e-4
        )

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) +
            list(self.critic2.parameters()),
            lr=3e-4
        )

        self.gamma = 0.99
        self.tau = 0.005

    # ==========================================
    # Action
    # ==========================================

    def act(self, s):

        s = torch.FloatTensor(s).unsqueeze(0).to(device)

        with torch.no_grad():

            z = self.encoder(s)
            a = self.actor(z)

        return a.cpu().numpy()[0]

    # ==========================================
    # Train Step
    # ==========================================

    def train(self, buffer, batch=256):

        s, a, s2, r, d = buffer.sample(batch)

        # ----------------------------
        # Encode
        # ----------------------------

        z = self.encoder(s)
        z2 = self.encoder(s2)

        z_sa = self.sa_encoder(z, a)

        # ----------------------------
        # Representation Loss
        # ----------------------------

        pred_z2, pred_r, pred_d = self.dynamics(z_sa)

        loss_dyn = F.mse_loss(pred_z2, z2.detach())
        loss_r = F.mse_loss(pred_r, r)
        loss_d = F.mse_loss(pred_d, d)

        rep_loss = loss_dyn + loss_r + loss_d

        # ----------------------------
        # Critic Loss
        # ----------------------------

        with torch.no_grad():

            a2 = self.actor_target(z2)

            z2_sa = self.sa_encoder(z2, a2)

            q1_t = self.critic1_target(z2_sa)
            q2_t = self.critic2_target(z2_sa)

            q_target = r + self.gamma * (1 - d) * torch.min(q1_t, q2_t)

        q1 = self.critic1(z_sa)
        q2 = self.critic2(z_sa)

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # ----------------------------
        # Optimize Representation + Critic
        # ----------------------------

        self.opt.zero_grad()
        self.critic_opt.zero_grad()

        (rep_loss + critic_loss).backward()

        self.opt.step()
        self.critic_opt.step()

        # ----------------------------
        # Actor
        # ----------------------------

        a_pi = self.actor(z)
        z_sa_pi = self.sa_encoder(z, a_pi)

        actor_loss = -self.critic1(z_sa_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ----------------------------
        # Target Update
        # ----------------------------

        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)