class ReplayBuffer:

    def __init__(
        self,
        obs_shape,
        action_dim,
        capacity: int,
        nstep: int = 1,
    ):
        pass

    def add(self, obs, action, reward, next_obs, done):
        pass

    def sample(self, batch_size: int) -> dict:
        """
        Returns:
            {
                "obs": Tensor [B, T, ...]
                "actions": Tensor [B, T, A]
                "rewards": Tensor [B, T, 1]
                "dones": Tensor [B, T, 1]
            }
        """
        pass