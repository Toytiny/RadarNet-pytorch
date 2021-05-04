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


def detection_cls_loss(car_det,moc_det,loss_label_car,loss_label_moc):
    loss_car_cls=0
    loss_moc_cls=0
    num_sample_car=0
    num_sample_moc=0
    if loss_label_car.size()[1]==0:
        loss_car_cls=0
    else:
        for i in range(loss_label_car.size()[0]):
            if torch.max(loss_label_car[i,:])>-1:
                num_sample_car+=1
                loss_car_cls-=torch.max(loss_label_car[i,:])*torch.log(car_det[i,0])+(\
                            1-torch.max(loss_label_car[i,:]))*torch.log(1-car_det[i,0])
                
        loss_car_cls=loss_car_cls/(num_sample_car+1e-5)
    
    if loss_label_moc.size()[1]==0:
        loss_moc_cls=0
    else:    
        for i in range(loss_label_moc.size()[0]):
            if torch.max(loss_label_moc.size()[0]):
                num_sample_moc+=1
                loss_moc_cls-=torch.max(loss_label_moc[i,:])*torch.log(moc_det[i,0])+(\
                            1-torch.max(loss_label_moc[i,:]))*torch.log(1-moc_det[i,0])
                
        loss_moc_cls=loss_moc_cls/(num_sample_moc+1e-5)
    
    return loss_moc_cls+loss_car_cls

def detection_reg_loss(car_det,moc_det,loss_label_car,loss_label_moc,gt_car,gt_moc):
    loss_car_reg=0
    loss_moc_reg=0
    num_positive_car=0
    num_positive_moc=0
    if loss_label_car.size()[1]==0:
        loss_car_reg=0
    else:
        for j in range(loss_label_car.size()[1]):
            for i in range(loss_label_car.size()[0]):
                if loss_label_car[i,j]>0:  # only on positive sample
                    num_positive_car+=1
                    for k in range(0,5):
                        if torch.abs(car_det[i,k+1]-gt_car[j,k])<1:  # smooth L1 loss on x, y, w, l, theta
                            loss_car_reg+=0.5*torch.pow(car_det[i,k+1]-gt_car[j,k],2)
                        else:
                            loss_car_reg+=abs(car_det[i,k+1]-gt_car[j,k])-0.5
        loss_car_reg=loss_car_reg/(num_positive_car+1e-5)
    
    if loss_label_moc.size()[1]==0:
        loss_moc_reg=0
    else:
        for j in range(loss_label_moc.size()[1]):
            for i in range(loss_label_moc.size()[0]):
                if loss_label_moc[i,j]>0:  # only on positive sample
                    num_positive_moc+=1
                    for k in range(0,5):
                        if torch.abs(moc_det[i,k+1]-gt_moc[j,k])<1:  # smooth L1 loss on x, y, w, l, theta
                            loss_moc_reg+=0.5*torch.pow(moc_det[i,k+1]-gt_moc[j,k],2)
                        else:
                            loss_moc_reg+=abs(moc_det[i,k+1]-gt_moc[j,k])-0.5
        loss_moc_reg=loss_moc_reg/(num_positive_moc+1e-5)
    
    return loss_moc_reg+loss_car_reg

def velocity_cls_loss(car_det,moc_det,loss_label_car,loss_label_moc,gt_car,gt_moc):
    loss_car_cls=0
    loss_moc_cls=0
    num_positive_car=0
    num_positive_moc=0
    if loss_label_car.size()[1]==0:
        loss_car_cls=0
    else:
        for j in range(loss_label_car.size()[1]):
            for i in range(loss_label_car.size()[0]):
                if loss_label_car[i,j]>0:  # only on positive sample
                    num_positive_car+=1
                    loss_car_cls-=gt_car[j,5]*torch.log(car_det[i,8])+(1-gt_car[j,5])*torch.log(1-car_det[i,8])
            
        loss_car_cls=loss_car_cls/(num_positive_car+1e-5)
        
    if loss_label_moc.size()[1]==0:
        loss_moc_cls=0
    else:
        for j in range(loss_label_moc.size()[1]):
            for i in range(loss_label_moc.size()[0]):
                if loss_label_moc[i,j]>0:  # only on positive sample
                    num_positive_moc+=1
                    loss_moc_cls-=gt_moc[j,5]*torch.log(moc_det[i,8])+(1-gt_moc[j,5])*torch.log(1-moc_det[i,8])
                
        loss_moc_cls=loss_moc_cls/(num_positive_moc+1e-5)
    
    
    return loss_moc_cls+loss_car_cls

def velocity_reg_loss(car_det,moc_det,loss_label_car,loss_label_moc,gt_car,gt_moc):
    loss_car_reg=0
    loss_moc_reg=0
    num_positive_car=0
    num_positive_moc=0
    if loss_label_car.size()[1]==0:
        loss_car_reg=0
    else:
        for j in range(loss_label_car.size()[1]):
            for i in range(loss_label_car.size()[0]):
                if loss_label_car[i,j]>0:  # only on positive sample
                    num_positive_car+=1
                    for k in range(5,7):
                        if torch.abs(car_det[i,k+1]-gt_car[j,k])<1:  # smooth L1 loss on v_x,v_y
                            loss_car_reg+=0.5*torch.pow(car_det[i,k+1]-gt_car[j,k],2)
                        else:
                            loss_car_reg+=abs(car_det[i,k+1]-gt_car[j,k])-0.5
        loss_car_reg=loss_car_reg/(num_positive_car+1e-5)
    
    if loss_label_moc.size()[1]==0:
        loss_moc_reg=0
    else:
        for j in range(loss_label_moc.size()[1]):
            for i in range(loss_label_moc.size()[0]):
                if loss_label_moc[i,j]>0:  # only on positive sample
                    num_positive_moc+=1
                    for k in range(5,7):
                        if torch.abs(moc_det[i,k+1]-gt_moc[j,k])<1:  # smooth L1 loss on v_x,v_y
                            loss_moc_reg+=0.5*torch.pow(moc_det[i,k+1]-gt_moc[j,k],2)
                        else:
                            loss_moc_reg+=abs(moc_det[i,k+1]-gt_moc[j,k])-0.5
        loss_moc_reg=loss_moc_reg/(num_positive_moc+1e-5)
    
    return loss_moc_reg+loss_car_reg

def refined_reg_loss(refined_vels_car,refined_vels_moc,loss_label_car,loss_label_moc,gt_car,gt_moc):
    loss_car_reg=0
    loss_moc_reg=0
    num_positive_car=0
    num_positive_moc=0
    if loss_label_car.size()[1]==0:
        loss_car_reg=0
    else:
        for j in range(loss_label_car.size()[1]):
            for i in range(loss_label_car.size()[0]):
                if loss_label_car[i,j]>0:  # only on true positive sample
                    num_positive_car+=1
                    for k in range(0,2):
                        if torch.abs(refined_vels_car[i,k]-gt_car[j,k])<1:  # smooth L1 loss on v_x,v_y
                            loss_car_reg+=0.5*torch.pow(refined_vels_car[i,k]-gt_car[j,k],2)
                        else:
                            loss_car_reg+=abs(refined_vels_car[i,k]-gt_car[j,k])-0.5
        loss_car_reg=loss_car_reg/(num_positive_car+1e-5)
    
    if loss_label_moc.size()[1]==0:
        loss_moc_reg=0
    else:
        for j in range(loss_label_moc.size()[1]):
            for i in range(loss_label_moc.size()[0]):
                if loss_label_moc[i,j]>0:  # only on true positive sample
                    num_positive_moc+=1
                    for k in range(0,2):
                        if torch.abs(refined_vels_moc[i,k]-gt_moc[j,k])<1:  # smooth L1 loss on v_x,v_y
                            loss_moc_reg+=0.5*torch.pow(refined_vels_moc[i,k]-gt_moc[j,k],2)
                        else:
                            loss_moc_reg+=abs(refined_vels_moc[i,k]-gt_moc[j,k])-0.5
        loss_moc_reg=loss_moc_reg/(num_positive_moc+1e-5)
    
    return loss_moc_reg+loss_car_reg