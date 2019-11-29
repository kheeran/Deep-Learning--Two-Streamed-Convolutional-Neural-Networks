import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn # Backend for using NVIDIA CUDA
import numpy as np

from torch import nn, optim
from torch.nn import functional as F
from torch.optim .optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter


# Enable benchmark mode on CUDNN since the input sizes do not vary. This finds the best algorithm to implement the convolutions given the layout.
torch.backends.cudnn.benchmark = True

# # maybe add parser?
# import argparse
# from pathlib import Path

class DataShape(NamedTuple):
    height: int
    width: int
    channels_xxx: int

# Use GPU if cuda is available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print ("Using CUDA...")
else:
    DEVICE = torch.device("cpu")
    print ("Using CPU...")

# The model class
class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels_xxx:int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = DataShape(height=height, width=width)
        self.class_count = class_count

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3,3),
            padding=(1,1),
        )

        self.initialise_layer(self.conv1)

        self.bnorm1 = nn.BatchNorm2d(
            num_features=32
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3,3),
            padding=(1,1),
        )

        self.initialise_layer(self.conv2)

        self.bnorm2 = nn.BatchNorm2d(
            num_features=64
        )

        # Pooling Layer with stride to half the output
        self.pool2 = nn.MaxPool2d(
            kernel_size=(2,2),
            stride=(2,2),
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3,3),
            padding=(1,1),
        )

        self.bnorm3 = nn.BatchNorm2d(
            num_features=64
        )

        # Could use Max Pooling for the last layer, but probably more likely to be stride
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3,3),
            padding=(1,1),
            stride=(2,2),
        )

        self.bnorm4 = nn.BatchNorm2d(
            num_features=64
        )

        self.fc1 = nn.Linear(15488, 1024)
        self.initialise_layer(self.fc1)

        self.bnormfc1 = nn.BatchNorm1d(
            num_features = 1024
        )

        self.fcout = nn.Linear (1024, 10)

        self.initialise_layer(self.fcout)

    

    # Initialise weights using Kaiming
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
