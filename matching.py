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

def matching_boxes(anchor_points,gt_boxes,device):
    
    
    num_pre=anchor_points.size()[0]
    num_gt=gt_boxes.size()[0]
    match_labels=(torch.zeros((num_pre,num_gt))).to(device)
    
    # Positive sample--the anchor_points with gt_boxes IoU>0.9
    for i in range(0,num_pre):
        for j in range(0,num_gt):
            # assuming size and orientation of gt_boxes
            int_pts = cv2.rotatedRectangleIntersection(((anchor_points[i,0], anchor_points[i,1]), (gt_boxes[j,2], gt_boxes[j,3]), \
                            gt_boxes[j,4]*180/3.14), ((gt_boxes[j,0], gt_boxes[j,1]), (gt_boxes[j,2], gt_boxes[j,3]), \
                            gt_boxes[j,4]*180/3.14))[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area  = cv2.contourArea(order_pts)
                inter     = int_area * 1.0 / (2*gt_boxes[j,2]*gt_boxes[j,3]-int_area + 1e-5)  # compute IoU
            else:
                inter = 0
            if inter>0.7: # IoU>0.7
                match_labels[i,j]=1
                print(inter)
            if inter<0.3: # IoU<0.3
                match_labels[i,j]=-1
    
    # To avoid the anchor points which corresponds to more than one gt_box
    for i in range(0,num_pre):
        if torch.sum(match_labels[i,:])>1:
            for j in range(0,num_gt):
                min_range=1e5
                if match_labels[i,j]==1:
                    dis=torch.sqrt(torch.pow(anchor_points[i,0]-gt_boxes[j,0],2)+torch.pow(anchor_points[i,1]-gt_boxes[j,1],2))
                    if dis<min_range:
                        min_range=dis
                    else:
                        match_labels[i,j]=0
                        
                    
    return match_labels

def matching_tp_boxes(match_label,det,device):
    
    
    num_pre=match_label.size()[0]
    num_gt=match_label.size()[1]
    for i in range(0,num_pre):
        for j in range(0,num_gt):
            if det[i,0]>0.5 and match_label[i,j]==1:
                match_label[i,j]=1
            else:
                match_label[i,j]=0
            
    return match_label
        
    