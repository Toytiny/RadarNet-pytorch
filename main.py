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
from utils import NMS
import numpy as np
import cv2
 
    
def main(opt):
    
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.eval
   
  
    print(opt)
    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    logger = Logger(opt)
        
    print('Creating model...')
    
    DecNet=nn.Sequential(
            Backbone(86),
            Header()
            )
    MLPNet=MLP()
    
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
      
        for ind, (annos, input_voxel, input_target) in enumerate(train_loader):
            if ind >= num_iters:
                break
            
            # Through the network
            input_voxel = input_voxel.float() 
            
            center_x=input_voxel.size()[2]/2
            center_y=input_voxel.size()[1]/2
            
            input_voxel = Variable(input_voxel)
            cls_res, reg_res = DecNet(input_voxel)
            
            # Seperate per class
            cls_car=cls_res[0,0]
            cls_moc=cls_res[0,1]
            
            reg_car=reg_res[0,0:9]
            reg_moc=reg_res[0,9:18]
            
            # Add the voxel center value
            for i in range(0,reg_car.size()[0]):
                for j in range(0,reg_moc.size()[1]):
                    reg_car[0,i,j]+=j
                    reg_car[1,i,j]+=i
                    reg_moc[0,i,j]+=j
                    reg_moc[1,i,j]+=i
            
            # Prepare the input for NMS
            car_boxes=torch.cat((reg_car[:4],(torch.atan(reg_car[5]/reg_car[4])*180/3.14)\
                                 .reshape(1,reg_car.size()[1],reg_car.size()[2]),reg_car[7:9]),dim=0)
            moc_boxes=torch.cat((reg_moc[:4],(torch.atan(reg_moc[5]/reg_moc[4])*180/3.14)\
                                 .reshape(1,reg_moc.size()[1],reg_moc.size()[2]),reg_moc[7:9]),dim=0)
            
            car_boxes=car_boxes.reshape(car_boxes.size()[0],car_boxes.size()[1]*car_boxes.size()[2]).t()
            moc_boxes=moc_boxes.reshape(moc_boxes.size()[0],moc_boxes.size()[1]*moc_boxes.size()[2]).t()
            
            car_scores=cls_car.reshape(cls_car.size()[0]*cls_car.size()[1])
            moc_scores=cls_moc.reshape(cls_moc.size()[0]*cls_moc.size()[1])
            
            # Keep the top 200 and NMS
            car_boxes,car_scores=NMS(car_boxes,car_scores,0.5,200)
            moc_boxes,moc_scores=NMS(moc_boxes,moc_scores,0.5,200)
          
            # Early output detections 
            # c,x,y,w,l,theta,v_x,v_y
            car_det=torch.cat((car_scores.reshape(car_scores.size()[0],1),car_boxes),dim=1)
            moc_det=torch.cat((moc_scores.reshape(moc_scores.size()[0],1),moc_boxes),dim=1)
            
            # Radar_target x,l,v_r,m,dt
            radar_target=input_target.reshape(input_target.size()[1],input_target.size()[2])
            
            # Detection-Radar Association
            for k in range(0,car_det.size()[0]):
                
                car_asso=torch.zeros((radar_target.size()[0],10))
                car_scores=torch.zeros(radar_target.size()[0])
                
                
                car_asso[:,0]=car_det[k,3]
                car_asso[:,1]=car_det[k,4]
                car_asso[:,2]=torch.sqrt(car_det[k,6]*car_det[k,6]+car_det[k,7]*car_det[k,7])
                car_asso[:,3]=car_det[k,6]/car_asso[:,2]
                car_asso[:,4]=car_det[k,7]/car_asso[:,2]
                
                car_asso[:,5]=torch.cos(torch.atan((car_det[k,1]-center_x)/(center_y-car_det[k,2]))+\
                     torch.atan(car_det[k,6]/car_det[k,7]))
                car_asso[:,6]=car_det[k,1]-radar_target[:,0]
                car_asso[:,7]=car_det[k,2]-radar_target[:,1]
                car_asso[:,8]=radar_target[:,4]
                
                beta=torch.atan((center_y-car_det[k,2])/(car_det[k,1]-center_x))-\
                    torch.atan((center_y-radar_target[:,1])/(radar_target[:,0]-center_x))
                car_asso[:,9]=radar_target[:,2]/torch.cos(torch.acos(car_asso[:,5])+beta)
                for j in range(0,car_scores.size()[0]):
                    if car_asso[i,9]>50:
                        car_asso[i,9]=50
                    
                
                for j in range(0,car_scores.size()[0]):
                    car_scores[i]=MLPNet(car_asso[j,:])
                    
                    

                
                    
                
            
if __name__ == '__main__':
  opt = opts().parse()
  main(opt)

