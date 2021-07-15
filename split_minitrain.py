#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:51:53 2021

@author: toytiny
"""
import time 
import sys
import os
import shutil

data_path="/home/toytiny/Desktop/RadarNet/data/nuscenes/"
voxel_path=data_path+"voxel_representations/mini_train/"
annos_path=data_path+"annotations/mini_train/"
voxel_target=data_path+"voxel_representations/mini_val/"
annos_target=data_path+"annotations/mini_val/"
shutil.rmtree(voxel_target)
shutil.rmtree(annos_target)
os.mkdir(voxel_target)
os.mkdir(annos_target)
voxel_list=sorted(os.listdir(voxel_path),key=lambda x:eval(x.split("/")[-1].split("-")[-1].split(".")[0]))
annos_list=sorted(os.listdir(annos_path),key=lambda x:eval(x.split("/")[-1].split("-")[-1].split(".")[0]))
interval=6
length=len(voxel_list)
for i in range(0,length):
    idx=i+1
    if idx%interval==0:
        shutil.move(voxel_path+voxel_list[i],voxel_target+voxel_list[i])
        shutil.move(annos_path+annos_list[i],annos_target+annos_list[i])
        