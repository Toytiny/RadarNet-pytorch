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
from torch.autograd import Variable
from logger import Logger
import json
import ujson
from datasets.nuscenes import nuScenes
from backbone import Backbone
from header import Header
from mlp import MLP
from utils import NMS, softmax, late_fusion, get_detection_label
import numpy as np
import cv2
 
    
def main(opt):
    
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.eval
    use_gpu = torch.cuda.is_available()
  
    print(opt)

    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger = Logger(opt)
        
    print('Creating model...')
    
    BBNet=Backbone(86).to(device)
    Header_car=Header().to(device)
    Header_moc=Header().to(device)       
    MLPNet=MLP().to(device)
    
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
                
            # Through the network
            input_voxel = input_voxel.float() 
            
            center_x=input_voxel.size()[2]/2
            center_y=input_voxel.size()[1]/2
            
            input_target=input_target.to(device)
            input_voxel = Variable(input_voxel).to(device)
            
            backbone=BBNet(input_voxel)
            
            cls_car, reg_car = Header_car(backbone)
            cls_moc, reg_moc = Header_moc(backbone)
            
            cls_car=cls_car.reshape((cls_car.size()[1:4]))
            cls_moc=cls_moc.reshape((cls_moc.size()[1:4]))
            reg_car=reg_car.reshape((reg_car.size()[1:4]))
            reg_moc=reg_moc.reshape((reg_moc.size()[1:4]))
            
            # Add the voxel center value
            for i in range(0,reg_car.size()[0]):
                for j in range(0,reg_moc.size()[1]):
                    reg_car[0,i,j]+=j*4+2
                    reg_car[1,i,j]+=i*4+2
                    reg_moc[0,i,j]+=j*4+2
                    reg_moc[1,i,j]+=i*4+2
            
            
            # Prepare the input for NMS
            car_boxes=torch.cat((reg_car[:4],(torch.atan(reg_car[5]/reg_car[4])*180/3.14)\
                                 .reshape(1,reg_car.size()[1],reg_car.size()[2]),reg_car[7:9]),dim=0)
            moc_boxes=torch.cat((reg_moc[:4],(torch.atan(reg_moc[5]/reg_moc[4])*180/3.14)\
                                 .reshape(1,reg_moc.size()[1],reg_moc.size()[2]),reg_moc[7:9]),dim=0)
            
            car_boxes=car_boxes.reshape(car_boxes.size()[0],car_boxes.size()[1]*car_boxes.size()[2]).t()
            moc_boxes=moc_boxes.reshape(moc_boxes.size()[0],moc_boxes.size()[1]*moc_boxes.size()[2]).t()
            
            car_scores=cls_car.reshape(cls_car.size()[1]*cls_car.size()[2])
            moc_scores=cls_moc.reshape(cls_moc.size()[1]*cls_moc.size()[2])
            
            # Keep the top 200 and NMS
            car_boxes,car_scores=NMS(car_boxes,car_scores,0.5,200)
            moc_boxes,moc_scores=NMS(moc_boxes,moc_scores,0.5,200)
          
            # Early output detections 
            # c,x,y,w,l,theta,v_x,v_y
            car_det=torch.cat((car_scores.reshape(car_scores.size()[0],1),car_boxes),dim=1)
            moc_det=torch.cat((moc_scores.reshape(moc_scores.size()[0],1),moc_boxes),dim=1)
            
            # Radar_target x,y,v_r,m,dt
            radar_target=input_target.reshape(input_target.size()[1],input_target.size()[2])
            
            # Detection-Radar Association and velocity aggregation
            refined_vels_car=late_fusion(car_det,radar_target,center_x,center_y,MLPNet,device)
            refined_vels_moc=late_fusion(moc_det,radar_target,center_x,center_y,MLPNet,device)
            
            
            # Getting the label for car potitive detection
            det_label_car=get_detection_label(car_det,gt_car,device)
            det_label_moc=get_detection_label(moc_det,gt_moc,device)
            
            # Calculate the cross entropy loss for detection classification  
            loss_det_cls=-torch.sum(det_label_car*torch.log(car_det[:,0])+(1-det_label_car)* 
                        torch.log(1-car_det[:,0]))/car_det.size()[0]-torch.sum(det_label_moc*torch.log(moc_det[:,0]) 
                      +(1-det_label_moc)*torch.log(1-moc_det[:,0]))/moc_det.size()[0]
              
                
                
                
            
if __name__ == '__main__':
  opt = opts().parse()
  main(opt)

