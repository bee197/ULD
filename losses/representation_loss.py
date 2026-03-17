class RepresentationLoss:

    def __init__(
        self,
        lambda_r: float,
        lambda_d: float,
        lambda_dyn: float,
        H_enc: int,
    ):
        pass

    def compute(
        self,
        encoder,
        sa_encoder,
        dynamics,
        batch
    ) -> dict:
        """
        multi-step latent rollout

        Returns:
            {
                "loss": Tensor,
                "loss_dyn": Tensor,
                "loss_reward": Tensor,
                "loss_done": Tensor
            }
        """
        pass