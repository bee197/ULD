from torch.utils._config_typing import load_config

from trainers.trainer import Trainer


def main():

    cfg = load_config()

    trainer = Trainer(cfg)

    trainer.train()