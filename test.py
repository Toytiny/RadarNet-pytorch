from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from opts import opts
import torch
import torch.utils.data
from torch import nn

import torch.nn.functional as F
import numpy as np

from torchsummary import summary
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import json
import ujson
import shutil

from datasets.nuscenes import nuScenes
from backbone import Backbone
from header import Header
from mlp import MLP
from utils import NMS, softmax, late_fusion, output_process
from evaluate import evaluate_result
import numpy as np
from matching import matching_boxes,matching_tp_boxes
from loss import calculate_loss
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def draw_box(img,draw,res,color):
    

    x=res[0]
    y=res[1]
    length=res[2]
    width=res[3]
    # along the x-axis, anticlockwise direction;
    angle=-res[4]

    cosA = math.cos(angle)
    sinA = math.sin(angle)

    x1=x-0.5*length
    y1=y-0.5*width

    x0=x+0.5*length
    y0=y1

    x2=x1
    y2=y+0.5*width

    x3=x0
    y3=y2

    x0n= (x0 -x)*cosA -(y0 - y)*sinA + x
    y0n = (x0-x)*sinA + (y0 - y)*cosA + y

    x1n= (x1 -x)*cosA -(y1 - y)*sinA + x
    y1n = (x1-x)*sinA + (y1 - y)*cosA + y

    x2n= (x2 -x)*cosA -(y2 - y)*sinA + x
    y2n = (x2-x)*sinA + (y2 - y)*cosA + y

    x3n= (x3 -x)*cosA -(y3 - y)*sinA + x
    y3n = (x3-x)*sinA + (y3 - y)*cosA + y


    draw.line([(x0n, y0n),(x1n, y1n)], fill=color,width=2)
    draw.line([(x1n, y1n),(x2n, y2n)], fill=color,width=2)
    draw.line([(x2n, y2n),(x3n, y3n)],fill= color,width=2)
    draw.line([(x0n, y0n), (x3n, y3n)],fill=color,width=2)

    #plt.imshow(img)
    #plt.show()




def test(opt):
    
    #torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()
  
    device=torch.device("cuda:0" if use_gpu else "cpu")
    
    base_path="/home/toytiny/Desktop/RadarNet2/train_result/"
    
    data_path='/home/toytiny/Desktop/RadarNet/data/nuscenes/'
    
    test_path=base_path+"test.txt"
    
    with open(test_path, 'w') as f:
        f.write('This file records the test results\n')
        
        
    model_path=base_path+"model/"
    
    load_checkpoint=True
    
    visualization=True
    
    fig_path='/home/toytiny/Desktop/RadarNet2/figures/mini_val/'
    
    res_path='/home/toytiny/Desktop/RadarNet2/res_figures/'
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    model_files=sorted(os.listdir(model_path),key=lambda x:eval(x.split("-")[1].split(".")[0]))
    model_check=model_files[-1]
        
    print('Loading checkpoint model '+model_check,'for test')
        
    load_model=torch.load(model_path+model_check,map_location='cuda:0')
    #BBNet=Backbone(102).to(device)
    BBNet=load_model["backbone"]
    #Header_car=Header().to(device) 
    Header_car=load_model["header"]

    BBNet.eval()
    Header_car.eval()
    print('Setting up testing data...')
    test_loader = torch.utils.data.DataLoader(
            nuScenes(opt, opt.train_split, data_path), batch_size=1, 
            shuffle=True, num_workers=1, 
            pin_memory=True,drop_last=True)
    
    batch_size=4;
    iter_ind=0
    aver_ap=torch.zeros(1).to(device)
    iter_test=np.floor(len(test_loader)/batch_size)
    

    
    nms_thres=[0.05]
  
    max_keep=[200]
  
    for i in range(0,len(nms_thres)):
        for j in range(0,len(max_keep)):
          
            thres=nms_thres[i]
            max_det=max_keep[j]
    
    
            with torch.no_grad():
                
                for ind, (gt,voxel,filename) in enumerate(test_loader):
                    
                    
                    # whether to initilizate the iteration
                    if ind%batch_size==0:
                            
                        input_voxels=torch.Tensor([])
                        gt_cars=[]
                        iter_ind+=1
                        test_ap=torch.zeros(1).to(device)
                        filenames=[]
                        print('Starting iter-{} for testing'.format(int(iter_ind)))
                         
                    input_voxels=torch.cat([input_voxels,voxel],dim=0)
                    gt_cars.append(gt.reshape((gt.size()[1],gt.size()[2])))
                    filenames.append(filename)
                        # whether to continue
                    if not (ind+1)%batch_size==0:
                        continue
                        
                    else:
                            
                        # Through the network
                        input_voxels = input_voxels.float().to(device)
                        backbones=BBNet(input_voxels)
                        cls_cars, reg_cars=Header_car(backbones)
                        # c x y w l theta x_p y_p
                        car_dets= output_process(cls_cars,reg_cars,device,batch_size)
                            
                        for k in range(0,batch_size):
                                
                            gt_car=gt_cars[k].to(device)
                             
                            car_det=car_dets[k].t()
                                
                            car_output=NMS(car_det,thres,max_det)
                            
                            if visualization:
                                
                                filename=filenames[k]
                                bev_name='bev'+filename[0][4:-5]+'.jpg'
                                bev_path=fig_path+bev_name
                                save_path=res_path+bev_name
                                gt_all=gt_car.cpu()
                                pre_all=car_output.cpu()
                                
                        
                                img = Image.open(bev_path)
                                img = img.convert('RGB')
                                draw = ImageDraw.Draw(img)
                                
                                
                                
                                for i in range(0,gt_all.size()[0]):
                                    
                                     draw_box(img,draw,gt_all[i,:].numpy(),(0,255,0))
                                     
                                for j in range(0,pre_all.size()[0]):
                                    
                                     draw_box(img,draw,pre_all[j,1:].numpy(),(255,0,0))
                                     
                                img.save(save_path)    
                                    
                                    
                                    #p1=(gt_all[i,0:2]-gt_all[i,2:4]/2).numpy()
                                    #p2=(gt_all[i,0:2]+gt_all[i,2:4]/2).numpy()
                                    
                                    # p1=(gt_all[i,0:2]-4).numpy()
                                    # p2=(gt_all[i,0:2]+4).numpy()
                                    
                                    # cv2.rectangle(bev_fig, (int(p1[0]),int(p1[1])),\
                                    #               (int(p2[0]),int(p2[1])),(255,255,0), 2)
                                
                                # for j in range(0,car_output.size()[0]):
                                    
                                #     p1=(pre_all[j,1:3]-6).numpy()
                                #     p2=(pre_all[j,1:3]+6).numpy()
                                    
                                #     cv2.rectangle(bev_fig, (int(p1[0]),int(p1[1])),\
                                #                   (int(p2[0]),int(p2[1])),(0,255,255), 2)
                                #cv2.imshow('head', bev_fig)
                                #cv2.waitKey(0) 
                                #print(1)
                                
                                #cv2.imwrite(bev_path, bev_fig)   #save picture
                                
                                
                            
                            AP=evaluate_result(car_output,gt_car,device)
                            
                            test_ap=test_ap+AP
                                
                        test_ap=test_ap/batch_size   
                        ap_msg='The AP of iter-{} is {}'.format(iter_ind,test_ap.item())+'\n'
                        print(ap_msg)
                            
                    aver_ap=aver_ap+test_ap
                            
                aver_ap=aver_ap/iter_test 
                
                test_msg='AP for {} samples: {}   nms_thres={}  max_det={}\n'.format(len(test_loader),aver_ap.item(),thres,max_det)
                print(test_msg)
                with open(test_path,'a') as f:
                    f.write(test_msg)
                   
                

if __name__ == '__main__':
  
  opt = opts().parse()
  

  
  test(opt)
