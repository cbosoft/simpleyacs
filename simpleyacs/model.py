from typing import Literal

from torch import nn
from torchvision.models import resnet

from .config import CfgNode


class SimpleYacsModel(nn.Module):

    RESNET_SIZE = Literal[18, 34, 50, 101, 152]

    def __init__(self, n: RESNET_SIZE):
        super().__init__()
        resnet_func = getattr(resnet, f'resnet{n}')
        resnet_weights = getattr(resnet, f'ResNet{n}_Weights')
        self.resnet = resnet_func(weights=resnet_weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.Linear(128, 1),
        )

    @classmethod
    def from_config(cls, cfg: CfgNode):
        return cls(n=cfg.model.n)

    def forward(self, x):
        return self.resnet(x)

