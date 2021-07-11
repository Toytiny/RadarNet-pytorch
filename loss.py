import os
from opts import opts
import torch
import torch.utils.data
from torch import nn

import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from torch.autograd import Variable

import json
import ujson
from datasets.nuscenes import nuScenes
from backbone import Backbone
from header import Header

import numpy as np

def calculate_loss(match_label,det,gt_boxes,device):

    
    pre_scores=det[:,0]
   
    sum_loss=torch.zeros(1).to(device)

    loss_det_cls=torch.zeros(1).to(device)
    loss_det_reg=torch.zeros(1).to(device)
 
    num_gt=match_label.size()[1]
    num_pre=match_label.size()[0]
    
    num_p=torch.sum(match_label==1).item()
    
    #num_n=1*num_p
    
    cls_label=torch.amax(match_label,dim=1)
    
    index_p=torch.where(cls_label==1)[0]
    
    index_ns=torch.where(cls_label==-1)[0]
    
    loss_c=torch.zeros(num_pre).to(device) # loss used for mining
    loss_c[index_ns]=pre_scores[index_ns]
    loss_c,index_c=torch.sort(loss_c,descending=True)
    #index_n=index_c[0:num_n]  # index for the negative sample
    index_n=index_c
    # calculate the detection classification cross entropy loss
    #for ind in index_n:
    loss_det_cls-=torch.sum(torch.pow(pre_scores[index_n],2)*torch.log(1-pre_scores[index_n]+1e-10))
        
    #for ind in index_p: # loss for positive
    loss_det_cls-=torch.sum(torch.pow(1-pre_scores[index_p],2)*torch.log(pre_scores[index_p]+1e-10))
        
    #print('Calculating loss for',N,'positive samples')
    for k in range(0,num_gt):
        
       
        index_p=torch.where(match_label[:,k]==1)[0] # index for the postive sample
    
    
        # calculate the detection regression smooth L1 loss (only on positive sample)
        for ind in index_p:
   
            res_det=torch.zeros(5).to(device)
            det_a=torch.sqrt(torch.pow(det[ind,3],2)+torch.pow(det[ind,4],2))
            res_det[0]=(gt_boxes[k,0]-det[ind,1])/(det_a+1e-5)
            res_det[1]=(gt_boxes[k,1]-det[ind,2])/(det_a+1e-5)
            res_det[2]=torch.log(gt_boxes[k,2]/(det[ind,3]+1e-5))
            res_det[3]=torch.log(gt_boxes[k,3]/(det[ind,4]+1e-5))
            res_det[4]=torch.sin(gt_boxes[k,4]-det[ind,5])
            for i in range(0,5):
                if torch.abs(res_det[i])<1:  # smooth L1 loss on x, y, w, l, theta
                    loss_det_reg+=0.5*torch.pow(res_det[i],2)
                else:
                    loss_det_reg+=torch.abs(res_det[i])-0.5
            
      
        
    loss_det_cls=loss_det_cls/(num_p+1e-5)
    loss_det_reg=loss_det_reg/(num_p+1e-5)
   
    sum_loss=loss_det_cls+loss_det_reg
    #loss_msg=' loss_det_cls:{}\n loss_det_reg:{}\n loss_vel_cls:{}\n loss_vel_reg:{}\n loss_vel_att:{}\n'\
    #    .format(loss_det_cls.item(),loss_det_reg.item(),loss_vel_cls\
    #    .item(),loss_vel_reg.item(),loss_vel_att.item())
    # print(loss_msg)
    
    return sum_loss

