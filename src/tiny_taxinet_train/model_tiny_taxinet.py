import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torchvision import models

"""
    small model for tiny taxinet

    Follows the SISL Kochenderfer Lab architecture:

    Number of layers: 4
    Number of inputs: 128
    Number of outputs: 2
    Maximum layer size: 128

    2nd line: 128,16,8,8,2

    Input size: 128
    Layer 2: 16
    Layer 3: 8
    Layer 4: 8
    Output: 2

"""


class TinyTaxiNetDNN(nn.Module):
    def __init__(self, model_name="TinyTaxiNet", n_features_in: int = 128, hid: list = None):
        super(TinyTaxiNetDNN, self).__init__()

        if hid is None:
            hid = [16, 8, 8]
            logger.info("Using default hidden layer sizes: {}".format(hid))

        self.fc1 = torch.nn.Linear(n_features_in, hid[0])
        self.fc2 = torch.nn.Linear(hid[0], hid[1])
        self.fc3 = torch.nn.Linear(hid[1], hid[2])
        self.fc4 = torch.nn.Linear(hid[2], 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(z, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # remember no relu on last layer!
        x = self.fc4(x)

        return x
