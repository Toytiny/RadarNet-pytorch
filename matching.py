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
import cv2

def matching_boxes(anchor_points,gt_boxes,device):
    
    
   
    
    num_pre=anchor_points.size()[0]
    num_gt=gt_boxes.size()[0]
    match_labels=(torch.zeros((num_pre,num_gt))).to(device)
    
    # Positive sample--the anchor_points with gt_boxes IoU>0.6
    for j in range(0,num_gt):
        bb_anchor=anchor_points[:,0:2]-gt_boxes[j,2:4]/2
        bb_gt=torch.tensor([gt_boxes[j,0]-gt_boxes[j,2]/2, gt_boxes[j,1]-gt_boxes[j,3]/2,gt_boxes[j,2], gt_boxes[j,3]]).to(device)
        zl1=torch.logical_or(bb_anchor[:,0]>bb_gt[0]+bb_gt[2],bb_anchor[:,1]>bb_gt[1]+bb_gt[3])
        zl2=torch.logical_or(bb_anchor[:,0]+bb_gt[2]<bb_gt[0],bb_anchor[:,1]+bb_gt[3]<bb_gt[1])
        zl =torch.logical_or(zl1,zl2)
        nl =torch.logical_not(zl)
        index_zl=torch.where(zl)[0]
        index_nl=torch.where(nl)[0]
        match_labels[index_zl,j]=-1
        for ind in index_nl:
            inter=bb_overlap(bb_anchor[ind,:],bb_gt)
            
   
            if inter>0.6: # IoU>0.5
                match_labels[ind,j]=1
                #print(inter,j)
            if inter<0.2: # IoU<0.1
                match_labels[ind,j]=-1
    
    # To avoid the anchor points which corresponds to more than one gt_box
    mg=torch.sum(match_labels,dim=1)>1
    mg_index=torch.where(mg)[0]
    if mg_index.shape!=torch.Size([0]):
        for ind in mg_index:
            for j in range(0,num_gt):
                min_range=1e5
                if match_labels[ind,j]==1:
                    dis=torch.pow(anchor_points[ind,0]-gt_boxes[j,0],2)+torch.pow(anchor_points[ind,1]-gt_boxes[j,1],2)
                    if dis<min_range:
                        min_range=dis
                    else:
                        match_labels[ind,j]=0
                        
                    
    return match_labels

def matching_tp_boxes(match_label,det,device):
    
    
    num_pre=match_label.size()[0]
    num_gt=match_label.size()[1]
    tp_label=(torch.zeros((num_pre,num_gt))).to(device)
    for j in range(0,num_gt):
        tp=torch.logical_and(det[:,0]>0.5,match_label[:,j]==1)
        ntp=torch.logical_not(tp)
        tp_index=torch.where(tp)[0]
        ntp_index=torch.where(ntp)[0]
        tp_label[tp_index,j]=1
        tp_label[ntp_index,j]=0
        
         
            
    return tp_label
        
def bb_overlap(anchor,gt):
    
    
    [x1,y1]=anchor
    [x2,y2,w,h]=gt
    
    colInt = abs(min(x1 +w ,x2+w) - max(x1, x2))
    rowInt = abs(min(y1 + h, y2 +h) - max(y1, y2))
    overlap_area = colInt * rowInt
    area = w * h
    return overlap_area / (area*2 - overlap_area+1e-5)
