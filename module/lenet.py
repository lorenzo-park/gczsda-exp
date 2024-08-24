import torch
import torch.nn as nn
import torch.nn.functional as F

from module.etc import Flatten


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
        )

        self.num_features = 768

    def forward(self, x):
        return self.feature_extractor(x)
