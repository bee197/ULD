import yaml
from trainers.trainer import Trainer


def load_config(path="configs/config.yaml"):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    cfg = load_config()
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
