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

def get_positive_label(car_det,gt_car,device):
    
    if gt_car.size()[0]==0:
        loss_label=torch.zeros((car_det.size()[0],gt_car.size()[0]))-1
    else:    
        loss_label=torch.zeros((car_det.size()[0],gt_car.size()[0]))-1
        for i in range(0,car_det.size()[0]):
            inter_index=0
            max_inter=0
            for j in range(0,gt_car.size()[0]):
                area=car_det[i,3]*car_det[i,4]
                tarea=gt_car[j,2]*gt_car[j,3]
                int_pts = cv2.rotatedRectangleIntersection(((car_det[i,1], car_det[i,2]), (car_det[i,3], car_det[i,4]), \
                                car_det[i,5]), ((gt_car[j,0], gt_car[j,1]), (gt_car[j,2], gt_car[j,3]), gt_car[j,4]*180/3.14))[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)
                    int_area  = cv2.contourArea(order_pts)
                    inter     = int_area * 1.0 / (tarea + area - int_area + EPSILON)  # compute IoU 
                else:
                    inter = 0
                    
                if inter>max_inter:
                    max_inter=inter
                    inter_index=j
                    
            if max_inter>0.9:
                loss_label[i,inter_index]=1
            if max_inter<0.4:
                loss_label[i,inter_index]=0
            
    return loss_label.to(device)

def get_tp_label(car_det,gt_car,device):
    
    if gt_car.size()[0]==0:
        loss_label=torch.zeros((car_det.size()[0],gt_car.size()[0]))-1
    else:
        loss_label=torch.zeros((car_det.size()[0],gt_car.size()[0]))-1
        for i in range(0,car_det.size()[0]):
            inter_index=0
            max_inter=0
            for j in range(0,gt_car.size()[0]):
                area=car_det[i,3]*car_det[i,4]
                tarea=gt_car[j,2]*gt_car[j,3]
                int_pts = cv2.rotatedRectangleIntersection(((car_det[i,1], car_det[i,2]), (car_det[i,3], car_det[i,4]), \
                                car_det[i,5]), ((gt_car[j,0], gt_car[j,1]), (gt_car[j,2], gt_car[j,3]), gt_car[j,4]*180/3.14))[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)
                    int_area  = cv2.contourArea(order_pts)
                    inter     = int_area * 1.0 / (tarea + area - int_area + EPSILON)  # compute IoU 
                else:
                    inter = 0
                    
                if inter>max_inter:
                    max_inter=inter
                    inter_index=j
                    
            if max_inter>0.9 and car_det[i,8]>0.5:
                loss_label[i,inter_index]=1
            if max_inter<0.4:
                loss_label[i,inter_index]=0
            
    return loss_label.to(device)