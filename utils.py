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

    
    
    
    
    
def output_process(cls_car,reg_car,device,bz):


    
    
    # Add two channel to keep the anchor point image coordinate x_a, y_a
    reg_car=torch.cat((reg_car,torch.zeros((bz,2,reg_car.size()[2],reg_car.size()[3])).to(device)),axis=1)
   
    # Apply sigmoid for confidence score

    #reg_car[:,6]=torch.sigmoid(reg_car[:,6])


    # Add the voxel center value to get the predicted center 
    # and keep the anchor point BEV coordinate
    
    for i in range(0,reg_car.size()[2]):
        reg_car[:,7,i,:]=(i*4+2)
    for j in range(0,reg_car.size()[3]):
        reg_car[:,6,:,j]=(j*4+2)
            
    
    reg_car[:,0,:,:]+=reg_car[:,6,:,:]
    reg_car[:,1,:,:]+=reg_car[:,7,:,:]
          
           
    # getting theta through torch.atan2
    car_boxes=torch.cat((reg_car[:,:2],torch.abs(reg_car[:,2:4]),(torch.atan2(reg_car[:,5],reg_car[:,4]))\
                         .reshape(bz,1,reg_car.size()[2],reg_car.size()[3])\
                         ,reg_car[:,6:8]),dim=1)
   
    
    # Vectorize the output
    car_boxes=car_boxes.reshape(car_boxes.size()[0],car_boxes.size()[1],car_boxes.size()[2]*car_boxes.size()[3])
  
    
    car_scores=cls_car.reshape(cls_car.size()[0],cls_car.size()[1],cls_car.size()[2]*cls_car.size()[3])

    # Early output detections 
    # c,x,y,w,l,theta,m,v_x,v_y,x_a,y_a
    car_det=torch.cat((car_scores,car_boxes),dim=1)
    
    # for j in range(0,car_det.size()[1]):
    #     conf_nan=torch.where(torch.isnan(car_det[:,j]))[0]
   
    #     car_det[conf_nan,j]=0
 
    
    return car_det

def softmax(x):
    x_exp = torch.exp(x-max(x))
    x_sum = torch.sum(x_exp)
    s = x_exp / (x_sum+1e-5)    
    return s


    
def NMS(det,iou_threshold,max_detection):
    
    scores=det[:,0]
    boxes=det[:,1:]
    # keep the top max_detection results
    scores,indices= torch.sort(scores,descending=True)
    
    boxes=boxes[indices[:max_detection]]
    scores=scores[:max_detection]
    
    # scores=scores[torch.where(scores>0.3)[0]]
    #pos = 0             # a position index

    #N = max_detection  # number of input bounding boxes
    
    _,order = scores.sort(0, descending=True) 
    keep=[]
    x1 = boxes[:,0]-boxes[:,2]/2
    y1 = boxes[:,1]-boxes[:,3]/2
    x2 = boxes[:,0]+boxes[:,2]/2
    y2 = boxes[:,1]+boxes[:,3]/2
    w = boxes[:,2]
    h = boxes[:,3]
    areas = w*h
    
    while order.numel()>0:
        if order.numel()==1:
            i=order.item()
            keep.append(i)
            break
        else:
            i=order[0].item()
            keep.append(i)
            
        
        xx1 = x1[order[1:]].clamp(min=x1[i])   
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)  
        iou = inter / (areas[i]+areas[order[1:]]-inter)  
        idx = (iou <= iou_threshold).nonzero().squeeze() 
        if idx.numel() == 0:
            break
        order = order[idx+1]  
        
    output=torch.cat((scores[keep].reshape(len(keep),1),boxes[keep,:]),dim=1)  
    
    
    
    return output    
    
    # for i in range(N):

    #     #maxscore = scores[i]
    #     #maxpos   = i

    #     tbox   = boxes[i,:]    
    #     tscore = scores[i]

    #     pos = i + 1

    #     # get bounding box with maximum score
    #     #while pos < N:
    #     #    if maxscore < scores[pos]:
    #     #        maxscore = scores[pos]
    #     #        maxpos = pos
    #     #    pos = pos + 1

    #     # Add max score bounding box as a detection result
    #     #boxes[i,:] = boxes[maxpos,:]
    #     #scores[i]  = scores[maxpos]
    #     # swap i-th box with position of max box
    #     #boxes[maxpos,:] = tbox
    #     #scores[maxpos]  = tscore

    #     #tbox   = boxes[i,:]
    #     #tscore = scores[i]
    #     tarea  = tbox[2] * tbox[3]

    #     #pos = i + 1

    #     # NMS iterations, note that N changes if detection boxes fall below final_threshold
    #     while pos < N:
    #         box   = boxes[pos, :]
    #         score = scores[pos]
    #         area  = box[2] * box[3]
    #         try:
    #             int_pts = cv2.rotatedRectangleIntersection(((tbox[0], tbox[1]), (tbox[2], tbox[3]), tbox[4]*180/3.14), \
    #                                                        ((box[0], box[1]), (box[2], box[3]), box[4]*180/3.14))[1]
    #             if int_pts is not None:
    #                 order_pts = cv2.convexHull(int_pts, returnPoints=True)
    #                 int_area  = cv2.contourArea(order_pts)
    #                 inter     = int_area * 1.0 / (tarea + area - int_area + EPSILON)  # compute IoU
    #             else:
    #                 inter = 0
    #         except:
    #             """
    #               cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
    #               error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
    #             """
    #             inter = 0.9999

    #         if inter > iou_threshold:
    #             boxes[pos, :] = boxes[N-1, :]
    #             scores[pos]   = scores[N-1]
    #             N = N - 1
    #             pos = pos - 1 

    #         pos = pos + 1
    
    # output=torch.cat(scores[:N].reshape(scores.size()[0],1),boxes[:N],dim=1)        
    # return output

def late_fusion(car_det,match_label_car,radar_target,center_x,center_y,MLPNet,device):


    vel_det_att=torch.zeros((car_det.size()[0],2)).to(device)
    num_radar=30;
    car_asso=torch.zeros((car_det.size()[0],num_radar,10),dtype=torch.float).to(device)
    car_scores=torch.zeros((car_det.size()[0],num_radar+1),dtype=torch.float).to(device)
    #car_asso=torch.zeros((car_det.size()[0],radar_target.size()[0],10),dtype=torch.float).to(device)
    #car_scores=torch.zeros((car_det.size()[0],radar_target.size()[0]+1),dtype=torch.float).to(device)
    car_scores[:,-1]=1   
    
    num_tp=torch.sum(match_label_car==1).item()
    #print("Starting late fusion")
    
    index_tp=torch.where(torch.sum(match_label_car==1,axis=1)>0)[0] 
    
    for k in index_tp:
    #for k in range(0,car_det.size()[0]):
        
       # if torch.sum(tp_car_label[k,:]==1)>0:
        sd_dis=torch.sqrt(1e-5+torch.pow(radar_target[:,0]-car_det[k,1],2)+torch.pow(radar_target[:,1]-car_det[k,2],2))
        sd_dis,index_sd=torch.sort(sd_dis,descending=False)
        index_sd=index_sd[0:num_radar]  # index for the negative sample
                    
        car_asso[k,:,0]=car_det[k,3]
        car_asso[k,:,1]=car_det[k,4]
            
        car_vel=torch.sqrt(1e-5+car_det[k,7]*car_det[k,7]+car_det[k,8]*car_det[k,8])
        car_vel_x=car_det[k,7]/(car_vel+1e-5)
        car_vel_y=car_det[k,8]/(car_vel+1e-5)
            
        car_asso[k,:,2]=car_vel
        car_asso[k,:,3]=car_vel_x
        car_asso[k,:,4]=car_vel_y
            
        car_gamma=torch.cos(torch.atan((car_det[k,1]-center_x)/(center_y-car_det[k,2]+1e-5))+\
                         torch.atan(car_det[k,7]/(car_det[k,8]+1e-5)))
        car_asso[k,:,5]=car_gamma
        car_asso[k,:,6]=car_det[k,1]-radar_target[index_sd,0]
        car_asso[k,:,7]=car_det[k,2]-radar_target[index_sd,1]
        car_asso[k,:,8]=radar_target[index_sd,4]
        
        beta=torch.atan((center_y-car_det[k,2])/(car_det[k,1]-center_x+1e-5))-\
                        torch.atan((center_y-radar_target[index_sd,1])/(radar_target[index_sd,0]-center_x+1e-5))
                        
        car_vel_bp=radar_target[index_sd,2]/(torch.cos(torch.acos(car_gamma)+beta)+1e-5)
        car_asso[k,:,9]=car_vel_bp
            
        
            
        for j in range(0,car_scores.size()[1]-1):
            if car_asso[k,j,9]>320:
                car_asso[k,j,9]=320
                    
                    
    #for k in range(0,car_det.size()[0]):
        
       # if torch.sum(tp_car_label[k,:]==1)>0:  
    for k in index_tp:       
         for j in range(0,car_scores.size()[1]-1):             
             car_scores[k,j]=MLPNet(car_asso[k,j,:])
                      
            # Velocity Aggregation for car
            
         car_scores_norm=softmax(car_scores[k,:])
         velo_cand=torch.cat((car_vel_bp,car_vel.view(1)),0).t()
         mag_refined=torch.sum(car_scores_norm*velo_cand)
         velo_refined=mag_refined*torch.cat((car_vel_x.view(1),car_vel_y.view(1)),0)
                    
            
         vel_det_att[k,:]=velo_refined
     
    return vel_det_att 


