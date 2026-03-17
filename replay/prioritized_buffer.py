class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        alpha: float,
        beta: float,
    ):
        pass

    def sample(self, batch_size):
        pass

    def update_priority(self, indices, td_error):
        pass