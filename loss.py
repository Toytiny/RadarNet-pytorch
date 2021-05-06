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

import numpy as np
import cv2

def calculate_loss(match_label,tp_match_label,det,vel_att,gt_boxes,device):

    
    pre_scores=det[:,0]
   
    sum_loss=0
    loss_vel_reg=0
    loss_vel_cls=0
    loss_det_cls=0
    loss_det_reg=0
    loss_vel_att=0
    num_gt=match_label.size()[1]
    num_pre=match_label.size()[0]
    
    N=torch.sum(match_label==1).int()
   
    for k in range(0,num_gt):
        
        # Hard negative mining
        num_p=torch.sum(match_label[:,k]==1).int()
        num_n=3*num_p
        loss_c=torch.zeros(num_pre).to(device) # loss used for mining
        for p in range(0,num_pre):
            # calculate the confidence loss for negative samples, set 0 to other samples
            if match_label[p,k]==0:
                loss_c[p]=-torch.log(1-pre_scores[p])
            else:
                loss_c[p]=0
                
        loss_c,index_c=torch.sort(loss_c,descending=True)
        index_n=index_c[0:num_n]  # index for the negative sample
        index_p=torch.where(match_label[:,k]==1)[0] # index for the postive sample
        index_tp=torch.where(tp_match_label[:,k]==1)[0] # index for the true positive sample
        
        # calculate the detection classification cross entropy loss
        loss_det_cls+=torch.sum(loss_c.index_select(0,index_n)) # loss for negative
        for ind in index_p: # loss for positive
            loss_det_cls-=torch.log(pre_scores[ind])
        
        # calculate the detection regression smooth L1 loss (only on positive sample)
        for ind in index_p:
            for i in range(0,5):
                if torch.abs(det[ind,i+1]-gt_boxes[k,i])<1:  # smooth L1 loss on x, y, w, l, theta
                    loss_det_reg+=0.5*torch.pow(det[ind,i+1]-gt_boxes[k,i],2)
                else:
                    loss_det_reg+=abs(det[ind,i+1]-gt_boxes[k,i])-0.5
            
        # calculate the velocity classification cross entropy loss (only on positive sample)
        for ind in index_p:
            loss_vel_cls-=gt_boxes[k,7]*torch.log(det[ind,6])+(1-gt_boxes[k,7])*torch.log(1-det[ind,6])
            
        # calculate the velocity regression smooth L1 loss (only on positive sample)
        for ind in index_p:
            for i in range(5,7):
                if torch.abs(det[ind,i+2]-gt_boxes[k,i])<1:  # smooth L1 loss on v_x,v_y
                    loss_vel_reg+=0.5*torch.pow(det[ind,i+2]-gt_boxes[k,i],2)
                else:
                    loss_vel_reg+=abs(det[ind,i+2]-gt_boxes[k,i])-0.5
                    
        # calculate the refined velocity regression smooth L1 loss (only on true positive sample)
        for ind in index_tp:
            for i in range(5,7):
                if torch.abs(vel_att[ind,i-5]-gt_boxes[k,i])<1:  # smooth L1 loss on v_x,v_y
                    loss_vel_att+=0.5*torch.pow(vel_att[ind,i-5]-gt_boxes[k,i],2)
                else:
                    loss_vel_att+=abs(vel_att[ind,i-5]-gt_boxes[k,i])-0.5
        
    sum_loss+=0.1*(loss_vel_reg+loss_vel_cls)+loss_det_cls+loss_det_reg+0.1*loss_vel_att
    sum_loss=sum_loss/(N+1e-6)
    
    return sum_loss

