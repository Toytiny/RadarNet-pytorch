import time
import torch
from torch import nn, optim
import numpy as np
import sys
import os
import torch.nn.functional as F

from torchsummary import summary

def Conv3x3ReLU(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class Header(nn.Module):
    def __init__(self):
        super(Header, self).__init__()
        
        self.cls_layer=nn.Sequential(
            Conv3x3ReLU(in_channels=128, out_channels=128),
            Conv3x3ReLU(in_channels=128, out_channels=128),
            Conv3x3ReLU(in_channels=128, out_channels=128),
            Conv3x3ReLU(in_channels=128, out_channels=128),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.reg_layer=nn.Sequential(
            Conv3x3ReLU(in_channels=128, out_channels=128),
            Conv3x3ReLU(in_channels=128, out_channels=128),
            Conv3x3ReLU(in_channels=128, out_channels=128),
            Conv3x3ReLU(in_channels=128, out_channels=128),
            nn.Conv2d(in_channels=128, out_channels=6, kernel_size=3, stride=1, padding=1)
        )
    def forward(self,x):
        confs=self.cls_layer(x)
        locs=self.reg_layer(x)
         
        out=(confs,locs)
         
        return out
 


    
