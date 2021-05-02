from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from pyquaternion import Quaternion
import numpy as np
import torch
import json
import re
import ujson
import cv2
import os
import sys
import math
import copy
from tqdm import tqdm
import torch.utils.data.dataset as Dataset

class nuScenes(Dataset.Dataset):

  def __init__(self, opt, split, data_path):
      
    super(nuScenes, self).__init__()
    self.data_path = data_path
    self.split = split
    self.annos_path=self.data_path+'annotations/'+split+'/'
    self.voxel_path=self.data_path+'voxel_representations/'+split+'/'
    self.annos=sorted(os.listdir(self.annos_path),key=lambda x:eval(x.split("/")[-1].split("-")[-1].split(".")[0]))
    self.voxel=sorted(os.listdir(self.voxel_path),key=lambda x:eval(x.split("/")[-1].split("-")[-1].split(".")[0]))
    
  def __len__(self):
    return len(os.listdir(self.annos_path))


  def __getitem__(self, index):
      
    with open(self.annos_path+self.annos[index],'r') as f:
        annos=ujson.load(f)
    with open(self.voxel_path+self.voxel[index],'r') as f:
        voxel=ujson.load(f)
    
    radar_voxel=np.asarray(voxel['radar_feat'])
    input_voxel=radar_voxel
    for i in range(0,len(voxel['lidar_feat'])):
        current_lidar_voxel=np.asarray(voxel['lidar_feat'][i])
        input_voxel=np.concatenate((input_voxel,current_lidar_voxel),axis=0)
    
    radar_target=voxel['radar_target']
    
    input_target=np.zeros((len(radar_target),5))
    for i in range(0,len(radar_target)):
        input_target[i,:2]=radar_target[i]['location']
        input_target[i,2]=radar_target[i]['vel_r']
        input_target[i,3]=radar_target[i]['motion']
        input_target[i,4]=radar_target[i]['time']
        
    
    return annos, input_voxel, input_target
