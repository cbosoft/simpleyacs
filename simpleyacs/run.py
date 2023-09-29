from .config import get_config, finalise
from .train import Trainer


def run(experiment_file_name: str):
    cfg = get_config()
    cfg.merge_from_file(experiment_file_name)
    finalise(cfg)

    if cfg.action == 'train':
        trainer = Trainer.from_config(cfg)
        trainer.train()
    else:
        raise ValueError(f'Unknown action {cfg.action}')

