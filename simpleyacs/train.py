from tqdm import trange
import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import SimpleYacsModel
from .config import CfgNode
from .dataset import Dataset


class Trainer:

    def __init__(self, model: SimpleYacsModel, lr: float, n_epochs: int, batch_size: int, output_dir: str, ds: Dataset):
        self.model = model
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.output_dir = output_dir

        # only one dataset here, but you'd want to split into train/valid(/test if applicable)
        self.ds = ds

    @classmethod
    def from_config(cls, cfg: CfgNode):
        return cls(
            model=SimpleYacsModel.from_config(cfg),
            lr=cfg.training.lr,
            n_epochs=cfg.training.n_epochs,
            batch_size=cfg.training.batch_size,
            ds=Dataset.from_config(cfg),
            output_dir=cfg.output_dir,
        )

    def train(self):
        opt = torch.optim.Adam(self.model.parameters(), self.lr)
        loss_func = torch.nn.MSELoss()
        dl = DataLoader(self.ds, batch_size=self.batch_size)

        epochs = []
        losses = []
        # metrics, e.g. accuracy
        accuracies = []
        bar = trange(self.n_epochs, unit='epochs', desc='--')
        for epoch in bar:
            epoch_loss = 0.0
            for images, answers in dl:
                predictions = self.model(images)
                loss = loss_func(predictions, answers)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dl)
            bar.set_description(f'loss={epoch_loss:.1f}')

            if epoch % 5 == 0:
                self.model: torch.nn.Module
                torch.save(self.model.state_dict(), f'{self.output_dir}/state_at_{epoch=}.pt')

            # after the training epoch, you'd want to do validation too.
            # accuracy = ...

            # store data, to be save later
            epochs.append(epoch)
            losses.append(epoch_loss)

            # if validating...
            # accuracies.append(accuracy)

        # when training's complete, you probably want to save the training results
        np.savetxt(f'{self.output_dir}/losses.csv', np.array([epochs, losses]).T)

        # obviously, your run will have valid losses and metrics to store, too. Add valid loss above, and metrics below.
        # np.savetxt(f'{self.output_dir}/accuracy.csv', np.array([epochs, accuracy]).T)

