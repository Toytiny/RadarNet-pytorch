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
from time import *
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

def weights_init(m):
    
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)
    
def main(opt):
    
    #torch.manual_seed(opt.seed)
    num_chan=38
    batch_size=4;
    torch.backends.cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()
  
    device=torch.device("cuda:0" if use_gpu else "cpu")
    
    base_path="/home/toytiny/Desktop/RadarNet2/train_result/"
    
    data_path='/home/toytiny/Desktop/RadarNet/data/nuscenes/'
    
    loss_path=base_path+"loss.txt"
    model_path=base_path+"model/"
    
    load_checkpoint=False
    
  
        
        
    if not load_checkpoint:
        
        print('Creating model...')
    
        BBNet=Backbone(num_chan).to(device)
        BBNet.apply(weights_init)
        Header_car=Header().to(device) 
        Header_car.apply(weights_init)
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        with open(loss_path, 'w') as f:
            f.write('This file records the loss during the network training\n') 
        st_epoch=1
    
    else:
        
        model_files=sorted(os.listdir(model_path),key=lambda x:eval(x.split("-")[1].split(".")[0]))
        model_check=model_files[-1]
        
        print('Loading checkpoint model '+model_check)
        
        load_model=torch.load(model_path+model_check,map_location='cuda:0')
        BBNet=Backbone(num_chan).to(device)
        BBNet.load_state_dict(load_model["backbone"])
        Header_car=Header().to(device) 
        Header_car.load_state_dict(load_model["header"])
        st_epoch=len(model_files)+1
    
  
    
    print('Creating Optimizer...')
    
    optimizer_b = optim.Adam(BBNet.parameters(), lr=1e-2)
    optimizer_hc= optim.Adam(Header_car.parameters(), lr=1e-2)
 
    scheduler_b = torch.optim.lr_scheduler.StepLR(optimizer_b,step_size=5,gamma=0.8)
    scheduler_hc = torch.optim.lr_scheduler.StepLR(optimizer_hc,step_size=5,gamma=0.8)
   

    warmup_b = torch.optim.lr_scheduler.StepLR(optimizer_b,step_size=10,gamma=1.03)
    warmup_hc = torch.optim.lr_scheduler.StepLR(optimizer_hc,step_size=10,gamma=1.03)
  
    warmup_iter=0
    
  
    
    
    
   
    
    
    
 
    
    print('Setting up validation data...')
    val_loader = torch.utils.data.DataLoader(
            nuScenes(opt, opt.train_split, data_path), batch_size=1, 
            shuffle=False, num_workers=8, 
            pin_memory=True,drop_last=True)
    
    train_loader = torch.utils.data.DataLoader(
            nuScenes(opt, opt.train_split, data_path), batch_size=1, 
            shuffle=True, num_workers=8, 
            pin_memory=True, drop_last=True)
  
   
    ite_epoch=np.floor(len(train_loader)/batch_size)
    
    iter_ind=(st_epoch-1)*ite_epoch
    
    print('Batch_size is:',batch_size)
        

    torch.autograd.set_detect_anomaly(False)
   
    
    for epoch in range(st_epoch, opt.num_epochs + 1):
        
        # Training phase
        print('Starting training for epoch-{}'.format(epoch))
        

        
        print('Setting up train data...')
        train_loader = torch.utils.data.DataLoader(
            nuScenes(opt, opt.train_split, data_path), batch_size=1, 
            shuffle=True, num_workers=8, 
            pin_memory=True, drop_last=True)
        
        
        
        
        num_iters = len(train_loader)*opt.num_epochs/batch_size 
        loss_epoch=torch.zeros(1).to(device)
        ap_epoch=torch.zeros(1).to(device)
        
        
        BBNet.train()
        Header_car.train()
        
        # begin_time=time()
        inde=-1
        for ind, (gt,voxel,_) in enumerate(train_loader):
            
            if gt.size()[1]==0:
                continue
            inde+=1
            # whether to initilizate the iteration
            if inde%batch_size==0:
                
                input_voxels=torch.Tensor([])
                gt_cars=[]
                iter_ind+=1
                loss_all=torch.zeros(1).to(device)
                train_ap=torch.zeros(1).to(device)
                
                print('Starting iter-{} in epoch {}'.format(int(iter_ind),epoch))
            # begin_time=time()
            voxel=voxel.float()
            input_voxels=torch.cat([input_voxels,voxel],dim=0)
            gt_cars.append(gt.reshape((gt.size()[1],gt.size()[2])))
            
            # whether to continue
            if not (inde+1)%batch_size==0:
                continue
            
            else:
                
                # end_time=time()
                # runtime=end_time-begin_time
                # print('The time for data load is', runtime)
                # Through the network
                input_voxels = input_voxels.float().to(device)
               
        
                backbones=BBNet(input_voxels)
                
                if torch.any(torch.isnan(backbones)):
                    for parameter in BBNet.parameters():
                        print(torch.max(parameter))
                        
                    raise ValueError("backbone element is nan or inf")
                    
                
                cls_cars, reg_cars=Header_car(backbones)
                if torch.any(torch.isnan(reg_cars)):
                    raise ValueError("reg element is nan or inf")
                if torch.any(torch.isnan(cls_cars)):
                    raise ValueError("cls element is nan or inf")
                
                # c x y w l theta x_p y_p
                car_dets= output_process(cls_cars,reg_cars,device,batch_size)
                
                if torch.any(torch.isnan(car_dets)):
                    raise ValueError("det element is nan or inf")
                    
                for k in range(0,batch_size):
                    
                    gt_car=gt_cars[k].to(device)
                 
                    car_det=car_dets[k].t()
                    
                    #with torch.no_grad():
                    #    car_output=NMS(car_det,0.1,25600)
                    # Matching the predicted boxes with groundtruth boxes
                    # gt_car: x y w l theta vx vy
                    match_label_car=matching_boxes(car_det[:,6:8],gt_car[:,0:5],device)
                            
                    loss_car=calculate_loss(match_label_car,car_det,gt_car,device)
                    
                    loss_all=loss_all+loss_car
                    
                    with torch.no_grad():
                        
                        car_output=NMS(car_det,0.05,200)
                
                        AP=evaluate_result(car_output,gt_car,device)
                    
                        train_ap=train_ap+AP
            
                loss_all=loss_all/batch_size
                
                train_ap=train_ap/batch_size
                    
                loss_msg='The loss of iter-{} is {}'.format(iter_ind,loss_all.item())
                
                print(loss_msg)
                
                loss_epoch=loss_epoch+loss_all
                
                ap_epoch=ap_epoch+train_ap
                
                ap_msg='The AP of iter-{} is {}'.format(iter_ind,train_ap.item())+'\n'
                print(ap_msg)
                
                
                with open(loss_path,'a') as f:
                    f.write(loss_msg)
                    f.write(ap_msg)
                    
                
                optimizer_b.zero_grad()
                optimizer_hc.zero_grad()
                
                
                loss_all.backward()
               
                
                #nn.utils.clip_grad_norm_(BBNet.parameters(), max_norm=2, norm_type=2)
                #nn.utils.clip_grad_norm_(Header_car.parameters(), max_norm=2, norm_type=2)
                #nn.utils.clip_grad_norm_(MLPNet.parameters(), max_norm=2, norm_type=2) 
                
                if iter_ind<warmup_iter:
                    warmup_b.step()
                    warmup_hc.step()
                    
                
                optimizer_b.step()
                optimizer_hc.step()
                   
        ap_epoch=ap_epoch/ite_epoch
        loss_epoch=loss_epoch/ite_epoch
        e_ap='The AP of epoch-{} is {}'.format(epoch,ap_epoch.item())
        e_loss='The loss of epoch-{} is {}'.format(epoch,loss_epoch.item())+'\n'
        print(e_ap)
        print(e_loss)
        with open(loss_path,'a') as f:
            f.write(e_ap)
            f.write(e_loss)
            
          
        if iter_ind>warmup_iter:
            scheduler_b.step()
            scheduler_hc.step()
            #optimizer_hm.step()
            
        
        state={'backbone': BBNet.state_dict(), 
               'optimizer_b':optimizer_b.state_dict(), 
               'header':Header_car.state_dict(), 
               'optimizer_hc':optimizer_hc.state_dict(),  
               'epoch':epoch}
        
        current_path=model_path+'epoch-{}'.format(epoch)
        torch.save(state,current_path+'.pth')
        
        # validation phase
        print('Starting validation for epoch-{}'.format(epoch))
        
        BBNet.eval()
        Header_car.eval()
        
        car_det=torch.Tensor([])
        car_dets=torch.Tensor([])
       
        cls_cars=torch.Tensor([])
        input_voxels=torch.Tensor([])
        match_label_car=torch.Tensor([])
        reg_cars=torch.Tensor([])
        voxel=torch.Tensor([])
        
        # with torch.no_grad():
            
        #     aver_ap=torch.zeros(1).to(device)
            
        #     for ind, (gt_car,input_voxel,_) in enumerate(val_loader):
                
        #         if gt_car.size()[1]>0:
        #             gt_car=Variable(gt_car.reshape((gt_car.size()[1],gt_car.size()[2]))).to(device)
        #         else:
        #             continue
            
        #         input_voxel = input_voxel.float() 
   
        #         input_voxel = input_voxel.to(device)
                
        #         backbone=BBNet(input_voxel)
                
        #         cls_car, reg_car = Header_car(backbone)
        #         # cls_moc, reg_moc = Header_moc(backbone)

        #         # process the output from network
        #         car_det= output_process(cls_car,reg_car,device,batch_size)
        #         car_det=car_det.reshape((car_det.size()[1],car_det.size()[2])).t()
        #         car_output=NMS(car_det,0.1,200)
                
        #         AP=evaluate_result(car_output,gt_car,device)
                
        #         #print('current sample AP: {}'.format(AP))
                
        #         aver_ap+=AP
            
        #     aver_ap=aver_ap/len(val_loader)
            
        #     eval_msg='Epoch-{} Average Precision: {}\n'.format(epoch,aver_ap.item())
        #     print(eval_msg)
        #     with open(loss_path,'a') as f:
        #         f.write(eval_msg)
                

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
