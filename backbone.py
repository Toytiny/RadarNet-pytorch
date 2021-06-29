import time
import torch
from torch import nn, optim

import numpy as np
import sys
import os
import torch.nn.functional as F

from torchsummary import summary

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class Backbone(nn.Module):
    def __init__(self,in_c):
        super(Backbone, self).__init__()
     
        # initial Conv
        self.b0_1=BasicConv2d(in_c,32, kernel_size=3, stride=2, padding=1)
        self.b0_2=BasicConv2d(32,32,kernel_size=3,padding=1)
        self.b0_3=BasicConv2d(32,64,kernel_size=3, stride=2, padding=1)
     
        # first branch
        self.b1_0=BasicConv2d(64,32,kernel_size=3,padding=1)
        self.b1_1=nn.Sequential(
            BasicConv2d(32,32,kernel_size=3,padding=1),
            BasicConv2d(32,32,kernel_size=3,padding=1)
            )
        self.b1_2=BasicConv2d(64,32,kernel_size=1)
        self.b1_3=BasicConv2d(96,32,kernel_size=1)
        self.b1_4=nn.Sequential(
            BasicConv2d(32,32,kernel_size=3,padding=1),
            BasicConv2d(32,32,kernel_size=3,padding=1)
            )
        self.b1_5=BasicConv2d(64,32,kernel_size=1)
        self.b1_6=BasicConv2d(96,32,kernel_size=1)
        self.b1_7=nn.Sequential(
            BasicConv2d(32,32,kernel_size=3,padding=1),
            BasicConv2d(32,32,kernel_size=3,padding=1)
            )
        self.b1_8=BasicConv2d(64,32,kernel_size=1)
        self.b1_9=BasicConv2d(96,32,kernel_size=1)
        
        # second branch
        self.b2_0=BasicConv2d(64,64,kernel_size=3,stride=2,padding=1)
        self.b2_1=nn.Sequential(
            BasicConv2d(64,64,kernel_size=3,padding=1),
            BasicConv2d(64,64,kernel_size=3,padding=1)
            )
        self.b2_2=nn.Sequential(
            BasicConv2d(64,64,kernel_size=3,padding=1),
            BasicConv2d(64,64,kernel_size=3,padding=1)
            )
        self.b2_3=BasicConv2d(32,64,kernel_size=1)
        self.b2_4=BasicConv2d(96,64,kernel_size=1)
        
        self.b2_5=nn.Sequential(
            BasicConv2d(64,64,kernel_size=3,padding=1),
            BasicConv2d(64,64,kernel_size=3,padding=1)
            )
        self.b2_6=nn.Sequential(
            BasicConv2d(64,64,kernel_size=3,padding=1),
            BasicConv2d(64,64,kernel_size=3,padding=1)
            )
        
        self.b2_7=BasicConv2d(32,64,kernel_size=1)
        self.b2_8=BasicConv2d(96,64,kernel_size=1)
        self.b2_9=nn.Sequential(
            BasicConv2d(64,64,kernel_size=3,padding=1),
            BasicConv2d(64,64,kernel_size=3,padding=1)
            )
        self.b2_10=nn.Sequential(
            BasicConv2d(64,64,kernel_size=3,padding=1),
            BasicConv2d(64,64,kernel_size=3,padding=1)
            )
        
        self.b2_11=BasicConv2d(32,64,kernel_size=1)
        self.b2_12=BasicConv2d(96,64,kernel_size=1)
        
        # thrid branch
        self.b3_0=BasicConv2d(64,96,kernel_size=3,stride=2,padding=1)
        self.b3_1=nn.Sequential(
            BasicConv2d(96,96,kernel_size=3,padding=1),
            BasicConv2d(96,96,kernel_size=3,padding=1)
            )
        self.b3_2=nn.Sequential(
            BasicConv2d(96,96,kernel_size=3,padding=1),
            BasicConv2d(96,96,kernel_size=3,padding=1)
            )
        self.b3_3=nn.Sequential(
            BasicConv2d(96,96,kernel_size=3,padding=1),
            BasicConv2d(96,96,kernel_size=3,padding=1)
            )
        self.b3_4=BasicConv2d(32,96,kernel_size=1)
        self.b3_5=BasicConv2d(64,96,kernel_size=1)
        
        self.b3_6=nn.Sequential(
            BasicConv2d(96,96,kernel_size=3,padding=1),
            BasicConv2d(96,96,kernel_size=3,padding=1)
            )
        self.b3_7=nn.Sequential(
            BasicConv2d(96,96,kernel_size=3,padding=1),
            BasicConv2d(96,96,kernel_size=3,padding=1)
            )
        self.b3_8=nn.Sequential(
            BasicConv2d(96,96,kernel_size=3,padding=1),
            BasicConv2d(96,96,kernel_size=3,padding=1)
            )
        
        self.b3_9=BasicConv2d(32,96,kernel_size=1)
        self.b3_10=BasicConv2d(64,96,kernel_size=1)
        
        self.b3_11=nn.Sequential(
            BasicConv2d(96,96,kernel_size=3,padding=1),
            BasicConv2d(96,96,kernel_size=3,padding=1)
            )
        self.b3_12=nn.Sequential(
            BasicConv2d(96,96,kernel_size=3,padding=1),
            BasicConv2d(96,96,kernel_size=3,padding=1)
            )
        self.b3_13=nn.Sequential(
            BasicConv2d(96,96,kernel_size=3,padding=1),
            BasicConv2d(96,96,kernel_size=3,padding=1)
            )
        self.b3_14=BasicConv2d(32,96,kernel_size=1)
        self.b3_15=BasicConv2d(64,96,kernel_size=1)
        
        # FPN
        self.b4_1=BasicConv2d(32,128,kernel_size=3,padding=1)
        self.b4_2=BasicConv2d(64,128,kernel_size=3,padding=1)
        self.b4_3=BasicConv2d(96,128,kernel_size=3,padding=1)
        
        self.upsample1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.upsample2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        
    def forward(self,x):
        # first three Conv  
        p=self.b0_3(self.b0_2(self.b0_1(x)))
        
        # initial Conv in three branch
        s1=self.b1_0(p)
        s2=self.b2_0(p)
        s3=self.b3_0(s2)
        
        # first block
        o1=self.b1_1(s1)
        o2=self.b2_1(s2)+s2
        o2=self.b2_2(o2)+o2
        o3=self.b3_1(s3)+s3
        o3=self.b3_2(o3)+o3
        o3=self.b3_3(o3)+o3
           
        # first cross scale sum
        k1=o1+s1+F.interpolate(self.b1_2(o2),scale_factor=2,recompute_scale_factor=True)+F.interpolate(self.b1_3(o3),scale_factor=4,recompute_scale_factor=True)
        k2=o2+s2+F.interpolate(self.b2_3(o1),scale_factor=1/2,recompute_scale_factor=True)+F.interpolate(self.b2_4(o3),scale_factor=2,recompute_scale_factor=True)
        k3=o3+s3+F.interpolate(self.b3_4(o1),scale_factor=1/4,recompute_scale_factor=True)+F.interpolate(self.b3_5(o2),scale_factor=1/2,recompute_scale_factor=True)
        
        # second block
        f1=self.b1_4(k1)
        f2=self.b2_2(k2)+k2
        f2=self.b2_6(f2)+f2
        f3=self.b3_6(k3)+k3
        f3=self.b3_7(f3)+f3
        f3=self.b3_8(f3)+f3
        
        # second cross scale sum
        q1=f1+k1+F.interpolate(self.b1_5(f2),scale_factor=2,recompute_scale_factor=True)+F.interpolate(self.b1_6(f3),scale_factor=4,recompute_scale_factor=True)
        q2=f2+k2+F.interpolate(self.b2_7(f1),scale_factor=1/2,recompute_scale_factor=True)+F.interpolate(self.b2_8(f3),scale_factor=2,recompute_scale_factor=True)
        q3=f3+k3+F.interpolate(self.b3_9(f1),scale_factor=1/4,recompute_scale_factor=True)+F.interpolate(self.b3_10(f2),scale_factor=1/2,recompute_scale_factor=True)
        
        # third block
        g1=self.b1_7(q1)
        g2=self.b2_9(q2)+q2
        g2=self.b2_10(g2)+g2
        g3=self.b3_11(q3)+q3
        g3=self.b3_12(g3)+g3
        g3=self.b3_13(g3)+g3
        
        # third cross scale sum
        c1=g1+q1+F.interpolate(self.b1_8(g2),scale_factor=2,recompute_scale_factor=True)+F.interpolate(self.b1_9(g3),scale_factor=4,recompute_scale_factor=True)
        c2=g2+q2+F.interpolate(self.b2_11(g1),scale_factor=1/2,recompute_scale_factor=True)+F.interpolate(self.b2_12(g3),scale_factor=2,recompute_scale_factor=True)
        c3=g3+q3+F.interpolate(self.b3_14(g1),scale_factor=1/4,recompute_scale_factor=True)+F.interpolate(self.b3_15(g2),scale_factor=1/2,recompute_scale_factor=True)
        
        # FPN
        n=self.upsample1(self.b4_3(c3))
        n=self.upsample2(self.b4_2(c2)+n)
        n=self.b4_1(c1)+n
        
        return n
 
#def test():
    # net=Backbone(43)
    # net.cuda()
    # summary(net,(43,320,320))
    
#test()