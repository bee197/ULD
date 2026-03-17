from agents.actor import Actor
from agents.critic import TwinCritic
from models.dynamics import LatentDynamics
from models.encoder import StateEncoder
from models.sa_encoder import StateActionEncoder


class ULDAgent:

    def __init__(self, cfg):

        # encoders
        self.encoder: StateEncoder
        self.sa_encoder: StateActionEncoder

        # latent model
        self.dynamics: LatentDynamics

        # policy & value
        self.actor: Actor
        self.critic: TwinCritic

        # targets
        self.actor_target: Actor
        self.critic_target: TwinCritic

        # optimizers
        self.repr_opt
        self.critic_opt
        self.actor_opt

        # hyperparams
        self.gamma: float
        self.tau: float
        self.policy_delay: int
        self.H_Q: int
        self.H_enc: int

    def act(self, obs, eval_mode=False):
        pass

    def update(self, batch: dict, step: int) -> dict:
        """
        main training step
        """
        pass