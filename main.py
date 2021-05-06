from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from opts import opts
import torch
import torch.utils.data
from torch import nn

import torch.nn.functional as F
import numpy as np

from torchsummary import summary
import torch.optim as optim
from torch.autograd import Variable
from logger import Logger
import json
import ujson
from datasets.nuscenes import nuScenes
from backbone import Backbone
from header import Header
from mlp import MLP
from utils import NMS, softmax, late_fusion, output_process

import numpy as np
from matching import matching_boxes,matching_tp_boxes
from loss import calculate_loss
import cv2
 
    
def main(opt):
    
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.eval
    use_gpu = torch.cuda.is_available()
  
    print(opt)

    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger = Logger(opt)
        
    print('Creating model...')
    
    BBNet=Backbone(166).to(device)
    Header_car=Header().to(device)
    Header_moc=Header().to(device)       
    MLPNet=MLP().to(device)
    
    print('Creating Optimizer...')
    
    optimizer_b = optim.Adam(BBNet.parameters(), lr=1e-3)
    optimizer_hc= optim.Adam(Header_car.parameters(), lr=1e-3)
    optimizer_hm= optim.Adam(Header_moc.parameters(), lr=1e-3)
    optimizer_mlp=optim.Adam(MLPNet.parameters(),lr=1e-3)
    
    
    data_path="/home/toytiny/Desktop/RadarNet/data/nuscenes/"
    
    if opt.val_intervals < opt.num_epochs or opt.eval:
        print('Setting up validation data...')
        val_loader = torch.utils.data.DataLoader(
        nuScenes(opt, opt.val_split, data_path), batch_size=1, shuffle=False, 
              num_workers=0, pin_memory=True)
  
        print('Setting up train data...')
        train_loader = torch.utils.data.DataLoader(
            nuScenes(opt, opt.train_split, data_path), batch_size=1, 
            shuffle=False, num_workers=0, 
            pin_memory=True, drop_last=True)
        
    num_iters = len(train_loader) if opt.num_iters < 0 else opt.num_iters    
    

    
    print('Starting training...')
    
    for epoch in range(1, opt.num_epochs + 1):
      
        for ind, (gt_car,gt_moc,input_voxel, input_target) in enumerate(train_loader):
            if ind >= num_iters:
                break
       
          
        
            if gt_car.size()[1]>0:
                gt_car=gt_car.reshape((gt_car.size()[1],gt_car.size()[2])).to(device)
            else:
                gt_car=gt_car.reshape(0,0).to(device)
            if gt_moc.size()[1]>0:  
                gt_moc=gt_moc.reshape((gt_moc.size()[1],gt_moc.size()[2])).to(device)
            else:
                gt_moc=gt_moc.reshape(0,0).to(device)
                
            if not input_voxel.size()[1]==166:
                continue
            # Through the network
            input_voxel = input_voxel.float() 
            
            center_x=input_voxel.size()[2]/2
            center_y=input_voxel.size()[1]/2
            
            input_target=input_target.to(device)
            input_voxel = Variable(input_voxel).to(device)
            
            backbone=BBNet(input_voxel)
            
            cls_car, reg_car = Header_car(backbone)
            cls_moc, reg_moc = Header_moc(backbone)
            
            # process the output from network
            car_det,moc_det= output_process(cls_car,reg_car,cls_moc,reg_moc,device)
            
            # Matching the predicted boxes with groundtruth boxes
            match_label_car=matching_boxes(car_det[:,9:11],gt_car[:,0:5],device)
            match_label_moc=matching_boxes(moc_det[:,9:11],gt_moc[:,0:5],device)
            
            
            # Keep the top 200 and NMS
            #car_det=NMS(car_det,0.45,200)torch.autograd.set_detect_anomaly(True)
            #moc_det=NMS(moc_det,0.45,200)
          
            # Radar_target x,y,v_r,m,dt
            radar_target=input_target.reshape(input_target.size()[1],input_target.size()[2])
            
            # Getting the true positive detection
            tp_car_label=matching_tp_boxes(match_label_car,car_det,device)
            tp_moc_label=matching_tp_boxes(match_label_moc,moc_det,device)
            
            # Detection-Radar Association and velocity aggregation (only for true positive) 
            car_vel_att=late_fusion(car_det,tp_car_label,radar_target,center_x,center_y,MLPNet,device)
            moc_vel_att=late_fusion(moc_det,tp_moc_label,radar_target,center_x,center_y,MLPNet,device)
            
            # Calculte loss for detection and velocity (using hard negative mining)
            loss_car=calculate_loss(match_label_car,tp_car_label,car_det,car_vel_att,gt_car,device)
            loss_moc=calculate_loss(match_label_moc,tp_moc_label,moc_det,car_vel_att,gt_moc,device)
            
            loss_sum=loss_car+loss_moc
            
            # Back Propagation and parameter update
            optimizer_b.zero_grad()
            optimizer_hc.zero_grad()
            optimizer_hm.zero_grad()
            optimizer_mlp.zero_grad()
            
            loss_sum.backward()
            
            
            optimizer_b.step()
            optimizer_hc.step()
            optimizer_hm.step()
            optimizer_mlp.step()
            
if __name__ == '__main__':
  opt = opts().parse()
  main(opt)

