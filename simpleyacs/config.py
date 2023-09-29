import os
from datetime import datetime
from yacs.config import CfgNode


def get_config() -> CfgNode:

    cfg = CfgNode()

    # I like to include an 'action' key in my configs.
    # This lets you choose what to do with the model/data/hyperparams
    # for example, you might want to run a quick training to make sure
    # things are working ok before running in depth cross validation
    # using the same settings.
    cfg.action = 'train'

    # desribe the model
    cfg.model = CfgNode()

    # Here we only have one parameter relating to the model: n.
    cfg.model.n = 18 # Size of ResNet model. One of 18, 34, 50, 101, 152.

    # Parameters related to training
    cfg.training = CfgNode()

    cfg.training.lr = 1e-5

    # How many epochs?
    cfg.training.n_epochs = 5

    # Batch size - how many images to pass at once
    cfg.training.batch_size = 4

    # Data is stored here as we go.
    cfg.output_dir = 'training_results/{today}'

    return cfg


def finalise(cfg: CfgNode) -> CfgNode:
    today = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg.output_dir = cfg.output_dir.format(today=today)
    os.makedirs(cfg.output_dir)
    cfg.freeze()

    with open(f'{cfg.output_dir}/config.yaml', 'w') as f:
        cfg.dump(stream=f)
    return cfg
