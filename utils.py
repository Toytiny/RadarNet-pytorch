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


def output_process(cls_car,reg_car,cls_moc,reg_moc,device):

    cls_car=cls_car.reshape((cls_car.size()[1:4]))
    cls_moc=cls_moc.reshape((cls_moc.size()[1:4]))
    reg_car=reg_car.reshape((reg_car.size()[1:4]))
    reg_moc=reg_moc.reshape((reg_moc.size()[1:4]))
    
    # Add two channel to keep the anchor point image coordinate x_a, y_a
    reg_car=torch.cat((reg_car,torch.zeros((2,reg_car.size()[1],reg_car.size()[2])).to(device)),axis=0)
    reg_moc=torch.cat((reg_moc,torch.zeros((2,reg_moc.size()[1],reg_moc.size()[2])).to(device)),axis=0)
                      
    # Apply sigmoid for confidence score

    reg_car[6]=torch.sigmoid(reg_car[6])
    reg_moc[6]=torch.sigmoid(reg_moc[6])
    
    reg_car[0]=torch.exp(reg_car[0])
    reg_car[1]=torch.exp(reg_car[1])
    
    # Add the voxel center value to get the predicted center and keep the anchor point BEV coordinate
    for i in range(0,reg_car.size()[1]):
        for j in range(0,reg_car.size()[2]):
            reg_car[9,i,j]=(j*4+2)
            reg_car[10,i,j]=(i*4+2)
            reg_moc[9,i,j]=(j*4+2)
            reg_moc[10,i,j]=(i*4+2)
            reg_car[0,i,j]+=reg_car[9,i,j]
            reg_car[1,i,j]+=reg_car[10,i,j]
            reg_moc[0,i,j]+=reg_moc[9,i,j]
            reg_moc[1,i,j]+=reg_moc[10,i,j]
           
    reg_car[2]=torch.exp(reg_car[2])
    reg_car[3]=torch.exp(reg_car[3])
    
    # getting theta through torch.atan2
    car_boxes=torch.cat((reg_car[:4],(torch.atan2(reg_car[5],reg_car[4]))\
                         .reshape(1,reg_car.size()[1],reg_car.size()[2])\
                         ,reg_car[6:11]),dim=0)
    moc_boxes=torch.cat((reg_moc[:4],(torch.atan2(reg_moc[5],reg_moc[4]))\
                         .reshape(1,reg_moc.size()[1],reg_moc.size()[2])\
                         ,reg_moc[6:11]),dim=0)
    
    # Vectorize the output
    car_boxes=car_boxes.reshape(car_boxes.size()[0],car_boxes.size()[1]*car_boxes.size()[2]).t()
    moc_boxes=moc_boxes.reshape(moc_boxes.size()[0],moc_boxes.size()[1]*moc_boxes.size()[2]).t()
    
    car_scores=cls_car.reshape(cls_car.size()[1]*cls_car.size()[2])
    moc_scores=cls_moc.reshape(cls_moc.size()[1]*cls_moc.size()[2])
    
    # Early output detections 
    # c,x,y,w,l,theta,m,v_x,v_y,x_a,y_a
    car_det=torch.cat((car_scores.reshape(car_scores.size()[0],1),car_boxes),dim=1)
    moc_det=torch.cat((moc_scores.reshape(moc_scores.size()[0],1),moc_boxes),dim=1)
    
    return car_det,moc_det

def softmax(x):
    x_exp = torch.exp(x)
    x_sum = torch.sum(x_exp)
    s = x_exp / x_sum    
    return s

def NMS(det,iou_threshold,max_detection):
    
    scores=det[:,0]
    boxes=det[:,1:]
    # keep the top max_detection results
    scores,indices= torch.sort(scores,descending=True)
    
    boxes=boxes[indices[:max_detection]]
    scores=scores[:max_detection]
    pos = 0             # a position index

    N = max_detection  # number of input bounding boxes
    
    for i in range(N):

        maxscore = scores[i]
        maxpos   = i

        tbox   = boxes[i,:]    
        tscore = scores[i]

        pos = i + 1

        # get bounding box with maximum score
        while pos < N:
            if maxscore < scores[pos]:
                maxscore = scores[pos]
                maxpos = pos
            pos = pos + 1

        # Add max score bounding box as a detection result
        boxes[i,:] = boxes[maxpos,:]
        scores[i]  = scores[maxpos]
        # swap ith box with position of max box
        boxes[maxpos,:] = tbox
        scores[maxpos]  = tscore

        tbox   = boxes[i,:]
        tscore = scores[i]
        tarea  = tbox[2] * tbox[3]

        pos = i + 1

        # NMS iterations, note that N changes if detection boxes fall below final_threshold
        while pos < N:
            box   = boxes[pos, :]
            score = scores[pos]
            area  = box[2] * box[3]
            try:
                int_pts = cv2.rotatedRectangleIntersection(((tbox[0], tbox[1]), (tbox[2], tbox[3]), tbox[4]*180/3.14), \
                                                           ((box[0], box[1]), (box[2], box[3]), box[4]*180/3.14))[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)
                    int_area  = cv2.contourArea(order_pts)
                    inter     = int_area * 1.0 / (tarea + area - int_area + EPSILON)  # compute IoU
                else:
                    inter = 0
            except:
                """
                  cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                  error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                """
                inter = 0.9999

            if inter > iou_threshold:
                boxes[pos, :] = boxes[N-1, :]
                scores[pos]   = scores[N-1]
                N = N - 1
                pos = pos - 1 

            pos = pos + 1
            
    return det[:N]     

def late_fusion(car_det,tp_car_label,radar_target,center_x,center_y,MLPNet,device):


    vel_det_att=torch.zeros((car_det.size()[0],2)).to(device)
    car_asso=torch.zeros((car_det.size()[0],radar_target.size()[0],10),dtype=torch.float).to(device)
    car_scores=torch.zeros((car_det.size()[0],radar_target.size()[0]+1),dtype=torch.float).to(device)
    car_scores[:,-1]=1      
    
    for k in range(0,car_det.size()[0]):
        
        if torch.sum(tp_car_label[k,:]==1)>0:
                
                    
            car_asso[k,:,0]=car_det[k,3]
            car_asso[k,:,1]=car_det[k,4]
            
            car_vel=torch.sqrt(car_det[k,6]*car_det[k,6]+car_det[k,7]*car_det[k,7]).double()
            car_vel_x=car_det[k,6]/(car_vel+1e-5)
            car_vel_y=car_det[k,7]/(car_vel+1e-5)
            
            car_asso[k,:,2]=car_vel
            car_asso[k,:,3]=car_vel_x
            car_asso[k,:,4]=car_vel_y
            
            car_gamma=torch.cos(torch.atan((car_det[k,1]-center_x)/(center_y-car_det[k,2]))+\
                         torch.atan(car_det[k,6]/car_det[k,7]))
            car_asso[k,:,5]=car_gamma
            car_asso[k,:,6]=car_det[k,1]-radar_target[:,0]
            car_asso[k,:,7]=car_det[k,2]-radar_target[:,1]
            car_asso[k,:,8]=radar_target[:,4]
                    
            beta=torch.atan((center_y-car_det[k,2])/(car_det[k,1]-center_x))-\
                        torch.atan((center_y-radar_target[:,1])/(radar_target[:,0]-center_x))
                        
            car_vel_bp=radar_target[:,2]/(torch.cos(torch.acos(car_gamma)+beta)+1e-5)
            car_asso[k,:,9]=car_vel_bp
            
        
            
            for j in range(0,car_scores.size()[1]-1):
                if car_asso[k,j,9]>50:
                    car_asso[k,j,9]=50
                    
                    
    for k in range(0,car_det.size()[0]):
        
        if torch.sum(tp_car_label[k,:]==1)>0:  
            
            for j in range(0,car_scores.size()[1]-1):             
                car_scores[k,j]=MLPNet(car_asso[k,j,:])
                        
            # Velocity Aggregation for car
            
            car_scores_norm=softmax(car_scores[k,:])
            velo_cand=torch.cat((car_vel_bp,car_vel.view(1)),0).t()
            mag_refined=torch.sum(car_scores_norm*velo_cand)
            velo_refined=mag_refined*torch.cat((car_vel_x.view(1),car_vel_y.view(1)),0)
                    
            
            vel_det_att[k,:]=velo_refined
     
    return vel_det_att 


