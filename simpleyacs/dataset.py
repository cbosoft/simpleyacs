import torch
from torch.utils.data import Dataset as _Dataset


class Dataset(_Dataset):

    def __init__(self):
        self.n = 100
        self.answers = torch.randint(0, 2, (self.n,))
        size = (3, 64, 64)
        self.images = [
            (torch.zeros(size) + (torch.randint(0, 64, size) if a else torch.randint(191, 255, size))).float()/255.
            for a in self.answers
        ]
        self.answers = self.answers.float().unsqueeze(1)

    @classmethod
    def from_config(cls, cfg):
        # cfg is not used, but in a real application you'd need to build your dataset based on the contents of your config file. 
        return cls()

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        return self.images[i], self.answers[i]
