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

def softmax(x):
    x_exp = torch.exp(x)
    x_sum = torch.sum(x_exp)
    s = x_exp / x_sum    
    return s

def NMS(boxes,scores,iou_threshold,max_detection):
    
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
            
    return boxes[:N], scores[:N]      

def late_fusion(car_det,radar_target,center_x,center_y,MLPNet,device):
    refined_vels_car=torch.zeros((car_det.size()[0],2))
        
    for k in range(0,car_det.size()[0]):
                
        car_asso=torch.zeros((radar_target.size()[0],10)).to(device)
        car_scores=torch.zeros(radar_target.size()[0]+1).to(device)
                
                
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
        for j in range(0,car_scores.size()[0]-1):
            if car_asso[j,9]>50:
                car_asso[j,9]=50
                    
                
        for j in range(0,car_scores.size()[0]-1):
            car_scores[j]=MLPNet(car_asso[j,:])
                    
        # Velocity Aggregation for car
        car_scores[-1]=1
        car_scores_norm=softmax(car_scores)
        velo_cand=torch.cat((car_asso[:,9],car_asso[0,2].reshape(1))).t()
        mag_refined=torch.sum(car_scores*velo_cand)
        velo_refined=mag_refined*torch.cat((car_asso[0,3].reshape(1),car_asso[0,4].reshape(1)))
                
        refined_vels_car[k,:]=velo_refined
 
    return refined_vels_car         


