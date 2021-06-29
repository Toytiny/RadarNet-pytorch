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
    loss_vel_reg=torch.zeros(1).to(device)
    loss_vel_cls=torch.zeros(1).to(device)
    loss_det_cls=torch.zeros(1).to(device)
    loss_det_reg=torch.zeros(1).to(device)
    loss_vel_att=torch.zeros(1).to(device)
    num_gt=match_label.size()[1]
    num_pre=match_label.size()[0]
    
    N=torch.sum(match_label==1).item()
    
    #print('Calculating loss for',N,'positive samples')
    for k in range(0,num_gt):
        
        # Hard negative mining
        num_p=torch.sum(match_label[:,k]==1).int()
        num_n=2*num_p
        
        loss_c=torch.zeros(num_pre).to(device) # loss used for mining
        index_ns=torch.where(match_label[:,k]==-1)[0] 
        
        loss_c[index_ns]=pre_scores[index_ns]
        # for p in range(0,num_pre):
        #     # calculate the confidence loss for negative samples, set 0 to other samples
        #     if match_label[p,k]==-1:
        #         #loss_c[p]=-torch.log(1-pre_scores[p])
        #         loss_c[p]=pre_scores[p]
        #     else:
        #         loss_c[p]=0
                
        loss_c,index_c=torch.sort(loss_c,descending=True)
        index_n=index_c[0:num_n]  # index for the negative sample
        index_p=torch.where(match_label[:,k]==1)[0] # index for the postive sample
        #index_tp=torch.where(tp_match_label[:,k]==1)[0] # index for the true positive sample
        
        # calculate the detection classification cross entropy loss
        for ind in index_n:
            loss_det_cls-=torch.log(1-pre_scores[ind]+1e-10)
        
        for ind in index_p: # loss for positive
            loss_det_cls-=torch.log(pre_scores[ind]+1e-10)
        
        # calculate the detection regression smooth L1 loss (only on positive sample)
        for ind in index_p:
            # det_t=torch.zeros(5).to(device)
            # det_t[0]=(gt_boxes[k,0]-det[ind,1])/(det[ind,3]+1e-5)
            # det_t[1]=(gt_boxes[k,1]-det[ind,2])/(det[ind,4]+1e-5)
            # det_t[2]=torch.log(gt_boxes[k,2]/(det[ind,3]+1e-5))
            # det_t[3]=torch.log(gt_boxes[k,3]/(det[ind,4]+1e-5))
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
            
        # # calculate the velocity classification cross entropy loss (only on positive sample)
        # for ind in index_p:
        #     loss_vel_cls-=gt_boxes[k,7]*torch.log(det[ind,6]+1e-5)+(1-gt_boxes[k,7])*torch.log(1-det[ind,6]+1e-5)
        #     if det[ind,6].isnan() or det[ind,6]>1:
        #         raise ValueError("velocity confidence is nan or inf")
        # # calculate the velocity regression smooth L1 loss (only on positive sample)
        # for ind in index_p:
        #     res_vel=torch.zeros(2).to(device)
        #     res_vel[0]=(gt_boxes[k,5]-det[ind,7])
        #     res_vel[1]=(gt_boxes[k,6]-det[ind,8])
        #     for i in range(0,2):
        #         if torch.abs(res_vel[i])<1:  # smooth L1 loss on v_x,v_y
        #             loss_vel_reg+=0.5*torch.pow(res_vel[i],2)
        #         else:
        #             loss_vel_reg+=torch.abs(res_vel[i])-0.5
                    
        # # calculate the refined velocity regression smooth L1 loss (only on true positive sample)
        # for ind in index_p:
        #     res_att=torch.zeros(2).to(device)
        #     res_att[0]=(gt_boxes[k,5]-vel_att[ind,0])
        #     res_att[1]=(gt_boxes[k,6]-vel_att[ind,1])
        #     for i in range(0,1):
        #         if torch.abs(res_att[i])<1:  # smooth L1 loss on v_x,v_y
        #             loss_vel_att+=0.5*torch.pow(res_att[i],2)
        #         else:
        #             loss_vel_att+=torch.abs(res_att[i])-0.5
    
    # loss_vel_reg=loss_vel_reg/(N+1e-5)
    # loss_vel_cls=loss_vel_cls/(N+1e-5)
    loss_det_cls=loss_det_cls/(N+1e-5)
    loss_det_reg=loss_det_reg/(N+1e-5)
    # loss_vel_att=loss_vel_att/(N+1e-5)
    
    # sum_loss+=0.1*(loss_vel_reg+loss_vel_cls)+loss_det_cls+loss_det_reg+0.1*loss_vel_att
    sum_loss=loss_det_cls+0.1*loss_det_reg
    #loss_msg=' loss_det_cls:{}\n loss_det_reg:{}\n loss_vel_cls:{}\n loss_vel_reg:{}\n loss_vel_att:{}\n'\
    #    .format(loss_det_cls.item(),loss_det_reg.item(),loss_vel_cls\
    #    .item(),loss_vel_reg.item(),loss_vel_att.item())
    # print(loss_msg)
    
    return sum_loss

