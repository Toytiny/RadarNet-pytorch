#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:04:20 2021

@author: toytiny
"""

import time 
import sys
import os
import shutil

fig_path="/home/toytiny/Desktop/RadarNet2/figures/"
train_path=fig_path+"mini_train/"
val_path=fig_path+"mini_val/"

shutil.rmtree(val_path)

os.mkdir(val_path)

fig_list=sorted(os.listdir(train_path),key=lambda x:eval(x.split("/")[-1].split("-")[-1].split(".")[0]))

interval=6
length=len(fig_list)
for i in range(0,length):
    idx=i+1
    if idx%interval==0:
        shutil.move(train_path+fig_list[i],val_path+fig_list[i])

        