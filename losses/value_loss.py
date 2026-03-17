class ValueLoss:

    def __init__(
        self,
        gamma: float,
        H_Q: int,
        use_double_q: bool = True,
    ):
        pass

    def compute(
        self,
        encoder,
        sa_encoder,
        critic,
        critic_target,
        actor_target,
        batch
    ) -> dict:
        """
        multi-step return

        Returns:
            {
                "loss": Tensor,
                "q_target": Tensor
            }
        """
        pass