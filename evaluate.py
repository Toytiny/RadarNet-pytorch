#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:23:31 2021

@author: toytiny
"""

import tqdm
import torch
import numpy as np


def evaluate_result(output,gt,device):
    
    num_pre=output.size()[0]
    
    tp=[]
    
    for j in range(0,num_pre):
        
        dis=torch.sqrt(torch.pow(output[j,1]-gt[:,0],2)+torch.pow(output[j,2]-gt[:,1],2))
        if torch.any(dis<16):
            tp.append(1)
        else:
            tp.append(0)
            
    conf=output[:,0].cpu().numpy().tolist()
    
    
    
    p,r= [], []
    
    n_gt=gt.size()[0]
    
    
    fpc = (1-np.array(tp)).cumsum()
    
    tpc = (np.array(tp)).cumsum()
    
    # if not len(tpc)>n_gt:
    recall_curve = tpc / (n_gt + 1e-16)
    r.append(recall_curve[-1])
    
    precision_curve = tpc / (tpc + fpc)
    p.append(precision_curve[-1])
    # else:
    #     recall_curve=tpc[:n_gt]/(n_gt+1e-16)
    #     r.append(recall_curve[-1])
    
    #     precision_curve = tpc[:n_gt] / (tpc[:n_gt] + fpc[:n_gt])
    #     p.append(precision_curve[-1])
    
    ap=compute_ap(recall_curve, precision_curve)
        
    return ap
    


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    # 将小于某元素前面的所有元素置为该元素，如[11,3,5,8,6]，操作后为[11,  8,  8,  8,  6]
    # 原因是 对于每个recall值r，我们要计算出对应（r’ > r）的最大precision
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    # recall_curve列表是有局部相等的，如[0,0.1,0.1,0.1,0.2,0.2,0.5,0.5],
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec , 微积分定义方式求解，小矩形相加
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap




