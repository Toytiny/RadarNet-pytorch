import time
import torch
from torch import nn, optim
import numpy as np
import sys
import os
import torch.nn.functional as F

from torchsummary import summary


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(10,32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.LayerNorm(1),
            nn.ReLU()
            )
 
    def forward(self, x):
        out = self.linear(x)
        return out



    


