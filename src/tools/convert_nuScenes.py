# Copyright (c) Fangqiang Ding. All Rights Reserved

import os
import json
import ujson
import numpy as np
from concurrent import futures
import copy
from time import *
from joblib import Parallel, delayed
import multiprocessing
import sys
import concurrent.futures
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.eval.detection.utils import category_to_detection_name
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from pointcloud import LidarPointCloud2 as LidarPointCloud
from pointcloud import RadarPointCloud2 as RadarPointCloud
DATA_PATH = '/home/toytiny/Desktop/RadarNet/data/nuscenes/'
OUT_PATH_PC = DATA_PATH + 'voxel_representations/'
OUT_PATH_AN = DATA_PATH + 'annotations/'
SPLITS = {
          #'mini_val': 'v1.0-mini',
          'mini_train': 'v1.0-mini',
          #'train': 'v1.0-trainval',
          #'val': 'v1.0-trainval',
          #'test': 'v1.0-test',
          }
DEBUG = False
CATS = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
SENSOR_ID = {'LIDAR_TOP': 1,'RADAR_FRONT': 2, 'RADAR_FRONT_LEFT': 3, 
  'RADAR_FRONT_RIGHT': 4, 'RADAR_BACK_LEFT': 5, 
  'RADAR_BACK_RIGHT': 6}

RADAR_LIST = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 
  'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 
  'RADAR_BACK_RIGHT']

LIDAR_LIST=['LIDAR_TOP']

#Put lidar in front of radar
USED_SENSOR=['LIDAR_TOP', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 
  'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 
  'RADAR_BACK_RIGHT']
CATS = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
CAT_IDS = {v: i + 1 for i, v in enumerate(CATS)}
NUM_SWEEPS_LIDAR = 1
NUM_SWEEPS_RADAR = 6

#suffix1 = '_{}sweeps'.format(NUM_SWEEPS) if NUM_SWEEPS > 1 else ''
#OUT_PATH = OUT_PATH + suffix1 + '/'

CAT_IDS = {v: i + 1 for i, v in enumerate(CATS)}

ATTRIBUTE_TO_ID = {
  '': 0, 'cycle.with_rider' : 1, 'cycle.without_rider' : 2,
  'pedestrian.moving': 3, 'pedestrian.standing': 4, 
  'pedestrian.sitting_lying_down': 5,
  'vehicle.moving': 6, 'vehicle.parked': 7, 
  'vehicle.stopped': 8}
side_range=(-50, 50) 
fwd_range=(-50, 50)
height_range = (-3,5)
res_height=0.25
res_wl = 0.15625
num_features=int((height_range[1]-height_range[0])/res_height);
num_x=int((fwd_range[1]-fwd_range[0])/res_wl);
num_y=int((side_range[1]-side_range[0])/res_wl);


                        
def voxel_generate_lidar(points,side_range,fwd_range, height_range, res_wl, res_height):
    
    num_features=int((height_range[1]-height_range[0])/res_height);
    num_x=int((fwd_range[1]-fwd_range[0])/res_wl);
    num_y=int((side_range[1]-side_range[0])/res_wl);

    feature_list=[]
    
    x_points = points[0,:]
    y_points = points[1,:]
    z_points = points[2,:]
    
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    h_filt = np.logical_and((z_points > height_range[0]), (z_points < height_range[1]))
    filt = np.logical_and(np.logical_and(f_filt, s_filt),h_filt)
    indices = np.argwhere(filt).flatten()
    points=points[:,indices]
    
    # Parallel(n_jobs=2)(delayed(extract_one_layer)(i) for i in range(0,num_features))
    for i in range(0,num_features):
      
        current_height_range=(height_range[0]+i*res_height,height_range[0]+(i+1)*res_height)
       
        # FILTER - To return only indices of points within desired cube
        # Three filters for: Front-to-back, side-to-side, and height ranges
        # Note left side is positive y axis car coordinates
        
        x_points = points[0,:]
        y_points = points[1,:]
        z_points = points[2,:]
        
        h_filt = np.logical_and((z_points > current_height_range[0]), (z_points < current_height_range[1]))
        
        
        if not any(h_filt):
            feature= np.zeros([num_y, num_x],dtype=np.float16)
            feature=feature.tolist()
            feature_list.append(feature)
            continue
        
        indices = np.argwhere(h_filt).flatten()

        # KEEPERS
        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]
   

        # SHIFT to the BEV view
        x_img = x_points + fwd_range[1] 
        y_img = -(y_points + side_range[0])
        z_img = z_points
        
        feature= np.zeros([num_y, num_x],dtype=np.float16)
        
        # calculate the value for each voxel in the feature
        for j in range(0,num_x):
            
            x_filt=np.logical_and((x_img > j*res_wl), (x_img < (j+1)*res_wl))
            if not any(x_filt):
                feature[:,j]=0
                continue
            else:
                # abandon those points not in this column to save time 
                indices=np.argwhere(x_filt).flatten()
                x_img_cl=x_img[indices]
                y_img_cl=y_img[indices]
                z_img_cl=z_img[indices]
                
            for k in range(0,num_y):
                y_filt=np.logical_and((y_img_cl > k*res_wl), (y_img_cl < (k+1)*res_wl))
                if not any(y_filt):
                    feature[k,j]=0
                else:
                    # continue to abandon those not in this row, namely keep points in this voxel
                    indices=np.argwhere(y_filt).flatten()
                    x_img_v=x_img_cl[indices]
                    y_img_v=y_img_cl[indices]
                    z_img_v=z_img_cl[indices]
            
                    # get the center of current voxel
                    v_x=(j+1/2)*res_wl
                    v_y=(k+1/2)*res_wl
                    v_z=(i+1/2)*res_height+height_range[0]
                    feature[k,j]+=np.sum((1-abs(x_img_v-v_x)/(res_wl/2))* \
                        (1-abs(y_img_v-v_y)/(res_wl/2))*(1-abs(z_img_v-v_z)/(res_height/2)))
                    # for n in range(0,len(x_img_v)):
                    
                    #     feature[k,j]+=(1-abs(x_img_v[n]-v_x)/(res_wl/2))* \
                    #     (1-abs(y_img_v[n]-v_y)/(res_wl/2))*(1-abs(z_img_v[n]-v_z)/(res_height/2))
                        
        feature=feature.astype(np.float16)
        feature=feature.tolist()
        feature_list.append(feature)
        
    return feature_list

def voxel_generate_radar(points,side_range,fwd_range, res_wl):
    
    num_x=int((fwd_range[1]-fwd_range[0])/res_wl);
    num_y=int((side_range[1]-side_range[0])/res_wl);
    
    feature= np.zeros([num_y, num_x],dtype=np.float16)
        
    
    x_points = points[0,:]
    y_points = points[1,:]
    dy_prop  = points[3,:]
            
            
    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis car coordinates
        
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filt = np.logical_and(f_filt, s_filt)
        
    if not any(filt):
        return feature
        
    indices = np.argwhere(filt).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    dy_prop  = dy_prop[indices]
        

    # SHIFT to the BEV view
    x_img = x_points + fwd_range[1] 
    y_img = -(y_points + side_range[0])
    
     
        
    # calculate the value for each voxel in the feature
    for j in range(0,num_x):
        x_filt=np.logical_and((x_img > j*res_wl), (x_img < (j+1)*res_wl))
        if not any(x_filt):
            feature[:,j]=0
            continue
        else:
            # abandon those points not in this column to save time 
            indices=np.argwhere(x_filt).flatten()
            x_img_cl=x_img[indices]
            y_img_cl=y_img[indices]
            dy_prop_cl=dy_prop[indices]
            for k in range(0,num_y):
                y_filt=np.logical_and((y_img_cl > k*res_wl), (y_img_cl < (k+1)*res_wl))
                if not any(y_filt):
                    feature[k,j]=0
                else:
                    indices=np.argwhere(y_filt).flatten()
                    dy_prop_v=dy_prop_cl[indices]
                    if any(dy_prop_v==0) or any(dy_prop_v==2) or any(dy_prop_v==6):
                        feature[k,j]=1
                    else:
                        feature[k,j]=-1
                    
                        
        
    feature=feature.tolist()
        
        
    return feature

def get_radar_target(points,times, side_range,fwd_range,height_range,res_wl):
    
    x_points = points[0,:]
    y_points = points[1,:]
    z_points = points[2,:]
    dy_prop=points[3,:]
    vel_x = points[8,:]
    vel_y = points[9,:]
    times=times[0,:]   
            
    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis car coordinates
        
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    h_filt = np.logical_and((z_points > height_range[0]), (z_points < height_range[1]))
    filt = np.logical_and(np.logical_and(f_filt, s_filt),h_filt)
        
    indices = np.argwhere(filt).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    dy_prop=dy_prop[indices]
    times=times[indices]
    vel_x = vel_x[indices]
    vel_y = vel_y[indices]
    
    
    # SHIFT to the BEV view

    x_img = np.floor((x_points + fwd_range[1])/res_wl)
    y_img = np.floor(-(y_points + side_range[0])/res_wl)
    
    # get the motion information
    for i in range(0,len(dy_prop)):
        if dy_prop[i] in [0,2,6]:
            dy_prop[i]=1
        else:
            dy_prop[i]=0
         
    # Transfer to the velocity in BEV view
    vel_x=vel_x/res_wl
    vel_y=-vel_y/res_wl
     
    # Calculate the radical velocity towards the car in BEV view
    theta=np.arctan((x_img-(fwd_range[1]-fwd_range[0])/2)/(y_img-1e-4-(side_range[1]-side_range[0])/2))
    vel_r=(-vel_y*np.cos(theta)-vel_x*np.sin(theta))
    
    targets=[]
    # point cloud information 
    for i in range(0,len(x_points)):
        if np.isnan(vel_r[i]):
            vel_r[i]=0
        target = {'location': [x_img[i],y_img[i]],
                  'vel_r': vel_r[i],
                  'motion':dy_prop[i],
                  'time': times[i], 
                  }
        targets.append(target)
        
    return targets   
        
   
def point_exist_in_box(box,pcs): 
     return True
   
    

def main():
  if not os.path.exists(OUT_PATH_PC):
    os.mkdir(OUT_PATH_PC)
  # convert one spilt at one time  
  for split in SPLITS:
    
    data_path = DATA_PATH
    nusc = NuScenes(
      version=SPLITS[split], dataroot=data_path, verbose=True)
    out_path_pc = OUT_PATH_PC + split 
    out_path_an = OUT_PATH_AN + split
    if not os.path.exists(out_path_pc):
        os.makedirs(out_path_pc)
    if not os.path.exists(out_path_an):
        os.makedirs(out_path_an)
    categories_info = [{'name': CATS[i], 'id': i + 1} for i in range(len(CATS))]
    ret = {'pcs': [], 'annotations': [], 'categories': categories_info, 
           'scenes': [], 'attributes': ATTRIBUTE_TO_ID}
    
    num_scenes = 0
    num_pcs = 0
    num_anns = 0
    
    # A "sample" in nuScenes refers to a timestamp with 5 RADAR and 1 LIDAR keyframe.
    for sample in nusc.sample:
      scene_name = nusc.get('scene', sample['scene_token'])['name']
      if not (split in ['test']) and \
         not (scene_name in SCENE_SPLITS[split]):
         continue
      if sample['prev'] == '':
        print('scene_name', scene_name)
        num_scenes+= 1
        ret['scenes'].append({'id': num_scenes, 'file_name': scene_name})
        track_ids = {}
        # skip the first keyframe since it has no prev sweeps  
        continue
      
      # Load lidar points from files and transform them to car coordinate  
      pc_token = sample['data'][LIDAR_LIST[0]]
      pc_data = nusc.get('sample_data', pc_token)
      num_pcs += 1
      out_path_current=out_path_pc+'/'+'voxel_scenes-{}_pcs-{}'.format(num_scenes,num_pcs)
      out_path_annos=out_path_an+'/'+'anno_scenes-{}_pcs-{}.json'.format(num_scenes,num_pcs)
      if os.path.exists(out_path_current) and os.path.exists(out_path_annos):
           continue
      # Complex coordinate transform from Lidar to car
      sd_record = nusc.get('sample_data', pc_token)
      cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
      pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
      trans_matr = transform_matrix(
                 cs_record['translation'], Quaternion(cs_record['rotation']),
                 inverse=False)
      # velocity transform from car to global
      vel_global_from_car = transform_matrix(np.array([0,0,0]),
            Quaternion(pose_record['rotation']), inverse=False)
      
      
      print('Aggregating lidar data for sample:', num_pcs)   
      
      lidar_pcs, _ = LidarPointCloud.from_file_multisweep(nusc, 
                               sample, LIDAR_LIST[0], LIDAR_LIST[0], NUM_SWEEPS_LIDAR)
      
        ## Transform lidar point clound from sensor coordinate to car
      voxel_lidar=[]
      voxel_radar=[]
      for i in range(0, len(lidar_pcs)):
            lidar_pcs[i][:3, :] = trans_matr.dot(np.vstack((lidar_pcs[i][:3, :], 
                                   np.ones(lidar_pcs[i].shape[1]))))[:3, :]
        #     # Extract voxel presentations for a timestamp
            print('Extracting voxel representation for lidar sweep:', i+1) 
            begin_time=time()
            voxel_lidar.append(voxel_generate_lidar(lidar_pcs[i],side_range,fwd_range,height_range,res_wl,res_height))
            end_time=time()
            runtime=end_time-begin_time
            print(runtime)
          
   
      radar_pcs=[list([]) for i in range(0,NUM_SWEEPS_RADAR)]
      radar_times=[list([]) for i in range(0,NUM_SWEEPS_RADAR)]
      print('Aggregating radar data for sample:', num_pcs)  
      # Read radar points from files and tranpose them to car coordinate        
      for sensor_name in RADAR_LIST:
          
          current_radar_pcs, current_radar_times = RadarPointCloud.from_file_multisweep(nusc, 
                        sample, sensor_name, LIDAR_LIST[0], NUM_SWEEPS_RADAR)
          #print(len(current_radar_pcs))
          # Transpose radar point clound from lidar coordinate to car (in the above function, the velocity is automatically
          # transformed to the car coordinate)
          for i in range(0, len(current_radar_pcs)):
              current_radar_pcs[i][:3, :] = trans_matr.dot(np.vstack((current_radar_pcs[i][:3, :], 
                                    np.ones(current_radar_pcs[i].shape[1]))))[:3, :]
              # stack points from all five radar sensors
              if not len(radar_pcs[i]):
                  radar_pcs[i] = current_radar_pcs[i]
                  radar_times[i] = current_radar_times[i]
              else:
                  radar_pcs[i] = np.hstack((radar_pcs[i], current_radar_pcs[i]))    
                  radar_times[i] = np.hstack((radar_times[i], current_radar_times[i]))    
      radar_target=[]            
      for i in range(0,NUM_SWEEPS_RADAR): 
          #print('Extracting voxel representation for radar sweep:', i+1)  
          voxel_radar.append(voxel_generate_radar(radar_pcs[i],side_range,fwd_range,res_wl))
          #print('Getting radar target information from radar sweep', i+1)  
          radar_target=np.hstack((radar_target, get_radar_target(radar_pcs[i], radar_times[i], 
                                                                  side_range,fwd_range,height_range,res_wl)))
      #point cloud information 
      h_layers=int((height_range[1]-height_range[0])/res_height)
      num_chan=int(NUM_SWEEPS_RADAR+NUM_SWEEPS_LIDAR*h_layers)
      out_path_dir=out_path_pc+'/'+'voxel_scenes-{}_pcs-{}'.format(num_scenes,num_pcs)+'/'
      if not os.path.exists(out_path_dir):
        os.makedirs(out_path_dir)
      chan=0
      
      for ch in range(0,NUM_SWEEPS_RADAR):
         
          curr_img=np.array(voxel_radar[ch])
          curr_img=np.array(curr_img*127+128,dtype='uint8')# map [-1 1] to [1 255]
          curr_img_path=out_path_dir+'{}.jpg'.format(ch)
          cv2.imwrite(curr_img_path,curr_img)
          chan+=1
          # save the lidar data then
      for lid in range(0,NUM_SWEEPS_LIDAR):
          
          for lay in range(0,h_layers):
              curr_img=np.array(voxel_lidar[lid][lay])
              curr_img=np.array(curr_img*30,dtype='uint8')
              curr_img_path=out_path_dir+'{}.jpg'.format(chan)
              cv2.imwrite(curr_img_path,curr_img)
              chan+=1
      # pc_input = {'id': num_pcs,
      #              'scene_id': num_scenes,
      #              'scene_name': scene_name,
      #              'radar_feat': voxel_radar, 
      #              'lidar_feat': voxel_lidar,
      #              'radar_target': radar_target.tolist(),
      #              'timestap': sample['timestamp']/1e6,
      #              }
      radar_voxel_channel=len(voxel_radar)
      lidar_voxel_channel=len(voxel_lidar)
      print('Save {} voxel for {} sample in {} scene'.format(
          split, num_pcs, num_scenes))
      print('Lidar voxel channel: {}, Radar voxel channel: {}'.format(lidar_voxel_channel,radar_voxel_channel))
       #print('out_path', out_path_current)
      

      
      # ujson.dump(pc_input, open(out_path_current, 'w'))
      
      
      
      _,boxes,_ = nusc.get_sample_data(pc_token, box_vis_level=BoxVisibility.ANY)
     
      anns = []
      boxes_in=[]
      print('Aggregating annotations for sample:', num_pcs) 
      # Abandon boxes not in the detection region
      for box in boxes:
          #nusc.render_annotation(my_instance['first_annotation_token'])
          # Transform the boxes from sensor coordinate to car
          box.rotate(Quaternion(cs_record['rotation']))
          box.translate(cs_record['translation'])
          
          f_filt = np.logical_and(((box.center[0]-box.wlh[0]/2)>fwd_range[0]),((box.center[0]+box.wlh[0]/2)<fwd_range[1]))
          s_filt = np.logical_and(((box.center[1]-box.wlh[1]/2)>-side_range[1]),((box.center[1]+box.wlh[1]/2)<-side_range[0]))
          h_filt = np.logical_and(((box.center[2]-box.wlh[2]/2)>height_range[0]),((box.center[2]+box.wlh[2]/2)<height_range[1]))
          filt = np.logical_and(np.logical_and(f_filt, s_filt),h_filt)
          
    
          if filt:
              boxes_in.append(box)
      
      boxes_all=[]
      for box in boxes_in:
         exist_point=point_exist_in_box(box,lidar_pcs)
         if exist_point:
             boxes_all.append(box)
             
          
              
      for box in boxes_all:
          # Map the catergory to detection name
          det_name = category_to_detection_name(box.name)
          if det_name is None:
              continue
          num_anns += 1
          category_id = CAT_IDS[det_name]
          v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
          yaw = np.arctan2(v[1], v[0])
          vel = nusc.box_velocity(box.token)
          # get velocity in car coordinates
          vel = np.dot(np.linalg.inv(vel_global_from_car), 
              np.array([vel[0], vel[1], vel[2], 0], np.float32))
          vel=copy.deepcopy(vel[:2]) # only keep v_x, v_y
          center=copy.deepcopy(box.center[:2]) # only keep the p_x, p_y
          wl=copy.deepcopy(box.wlh[:2]) # only keep the width and length
          # project the object center, velocity and wl from car coordinate to BEV
          # SHIFT to BEV coordinate
          center[0] = np.floor((center[0] + fwd_range[1])/res_wl)
          center[1] = np.floor(-(center[1] + side_range[0])/res_wl)
          
          wl[0]=wl[0]/res_wl
          wl[1]=wl[1]/res_wl
          
          vel[0]=vel[0]/res_wl
          vel[1]=-vel[1]/res_wl
          
          if np.isnan(vel[0]) or np.isnan(vel[1]):
              vel[0]=0
              vel[1]=0
              
          sample_ann = nusc.get(
              'sample_annotation', box.token)
          instance_token = sample_ann['instance_token']
          if not (instance_token in track_ids):
              track_ids[instance_token] = len(track_ids) + 1
          attribute_tokens = sample_ann['attribute_tokens']
          attributes = [nusc.get('attribute', att_token)['name'] \
                        for att_token in attribute_tokens]
          att = '' if len(attributes) == 0 else attributes[0]
          if len(attributes) > 1:
              print(attributes)
              import pdb; pdb.set_trace()
          track_id = track_ids[instance_token]
          # annotations information 
          ann = {
              'id': num_anns,   # id for annotation instance 
              'pc_id': num_pcs,     # id for point cloud sample instance, the same as 'id': num_pcs in pcs
              'category_id': category_id,   # id for category of the object
              'dim': [wl[0],wl[1]],   # dimension in width, length 
              'location': [center[0], center[1]],   # object center location  
              'rotation_z': yaw,    # object pitch angle
              'track_id': track_id,     # id for track object
              'attributes': ATTRIBUTE_TO_ID[att],  # object attributes
              'velocity':[vel[0],vel[1]],  # object velocity in the car coordinate, BEV images
            }
          anns.append(ann)
          
      print('Save {} annos for {} sample in {} scene'.format(
              split, num_pcs, num_scenes))
      
      #print('out_path', out_path_annos)
      print('======')
      ujson.dump(anns, open(out_path_annos, 'w'))
          

SCENE_SPLITS = {
'train':
    ['scene-0001', 'scene-0002', 'scene-0004', 'scene-0005', 'scene-0006', 'scene-0007', 'scene-0008', 'scene-0009',
     'scene-0010', 'scene-0011', 'scene-0019', 'scene-0020', 'scene-0021', 'scene-0022', 'scene-0023', 'scene-0024',
     'scene-0025', 'scene-0026', 'scene-0027', 'scene-0028', 'scene-0029', 'scene-0030', 'scene-0031', 'scene-0032',
     'scene-0033', 'scene-0034', 'scene-0041', 'scene-0042', 'scene-0043', 'scene-0044', 'scene-0045', 'scene-0046',
     'scene-0047', 'scene-0048', 'scene-0049', 'scene-0050', 'scene-0051', 'scene-0052', 'scene-0053', 'scene-0054',
     'scene-0055', 'scene-0056', 'scene-0057', 'scene-0058', 'scene-0059', 'scene-0060', 'scene-0061', 'scene-0062',
     'scene-0063', 'scene-0064', 'scene-0065', 'scene-0066', 'scene-0067', 'scene-0068', 'scene-0069', 'scene-0070',
     'scene-0071', 'scene-0072', 'scene-0073', 'scene-0074', 'scene-0075', 'scene-0076', 'scene-0120', 'scene-0121',
     'scene-0122', 'scene-0123', 'scene-0124', 'scene-0125', 'scene-0126', 'scene-0127', 'scene-0128', 'scene-0129',
     'scene-0130', 'scene-0131', 'scene-0132', 'scene-0133', 'scene-0134', 'scene-0135', 'scene-0138', 'scene-0139',
     'scene-0149', 'scene-0150', 'scene-0151', 'scene-0152', 'scene-0154', 'scene-0155', 'scene-0157', 'scene-0158',
     'scene-0159', 'scene-0160', 'scene-0161', 'scene-0162', 'scene-0163', 'scene-0164', 'scene-0165', 'scene-0166',
     'scene-0167', 'scene-0168', 'scene-0170', 'scene-0171', 'scene-0172', 'scene-0173', 'scene-0174', 'scene-0175',
     'scene-0176', 'scene-0177', 'scene-0178', 'scene-0179', 'scene-0180', 'scene-0181', 'scene-0182', 'scene-0183',
     'scene-0184', 'scene-0185', 'scene-0187', 'scene-0188', 'scene-0190', 'scene-0191', 'scene-0192', 'scene-0193',
     'scene-0194', 'scene-0195', 'scene-0196', 'scene-0199', 'scene-0200', 'scene-0202', 'scene-0203', 'scene-0204',
     'scene-0206', 'scene-0207', 'scene-0208', 'scene-0209', 'scene-0210', 'scene-0211', 'scene-0212', 'scene-0213',
     'scene-0214', 'scene-0218', 'scene-0219', 'scene-0220', 'scene-0222', 'scene-0224', 'scene-0225', 'scene-0226',
     'scene-0227', 'scene-0228', 'scene-0229', 'scene-0230', 'scene-0231', 'scene-0232', 'scene-0233', 'scene-0234',
     'scene-0235', 'scene-0236', 'scene-0237', 'scene-0238', 'scene-0239', 'scene-0240', 'scene-0241', 'scene-0242',
     'scene-0243', 'scene-0244', 'scene-0245', 'scene-0246', 'scene-0247', 'scene-0248', 'scene-0249', 'scene-0250',
     'scene-0251', 'scene-0252', 'scene-0253', 'scene-0254', 'scene-0255', 'scene-0256', 'scene-0257', 'scene-0258',
     'scene-0259', 'scene-0260', 'scene-0261', 'scene-0262', 'scene-0263', 'scene-0264', 'scene-0283', 'scene-0284',
     'scene-0285', 'scene-0286', 'scene-0287', 'scene-0288', 'scene-0289', 'scene-0290', 'scene-0291', 'scene-0292',
     'scene-0293', 'scene-0294', 'scene-0295', 'scene-0296', 'scene-0297', 'scene-0298', 'scene-0299', 'scene-0300',
     'scene-0301', 'scene-0302', 'scene-0303', 'scene-0304', 'scene-0305', 'scene-0306', 'scene-0315', 'scene-0316',
     'scene-0317', 'scene-0318', 'scene-0321', 'scene-0323', 'scene-0324', 'scene-0328', 'scene-0347', 'scene-0348',
     'scene-0349', 'scene-0350', 'scene-0351', 'scene-0352', 'scene-0353', 'scene-0354', 'scene-0355', 'scene-0356',
     'scene-0357', 'scene-0358', 'scene-0359', 'scene-0360', 'scene-0361', 'scene-0362', 'scene-0363', 'scene-0364',
     'scene-0365', 'scene-0366', 'scene-0367', 'scene-0368', 'scene-0369', 'scene-0370', 'scene-0371', 'scene-0372',
     'scene-0373', 'scene-0374', 'scene-0375', 'scene-0376', 'scene-0377', 'scene-0378', 'scene-0379', 'scene-0380',
     'scene-0381', 'scene-0382', 'scene-0383', 'scene-0384', 'scene-0385', 'scene-0386', 'scene-0388', 'scene-0389',
     'scene-0390', 'scene-0391', 'scene-0392', 'scene-0393', 'scene-0394', 'scene-0395', 'scene-0396', 'scene-0397',
     'scene-0398', 'scene-0399', 'scene-0400', 'scene-0401', 'scene-0402', 'scene-0403', 'scene-0405', 'scene-0406',
     'scene-0407', 'scene-0408', 'scene-0410', 'scene-0411', 'scene-0412', 'scene-0413', 'scene-0414', 'scene-0415',
     'scene-0416', 'scene-0417', 'scene-0418', 'scene-0419', 'scene-0420', 'scene-0421', 'scene-0422', 'scene-0423',
     'scene-0424', 'scene-0425', 'scene-0426', 'scene-0427', 'scene-0428', 'scene-0429', 'scene-0430', 'scene-0431',
     'scene-0432', 'scene-0433', 'scene-0434', 'scene-0435', 'scene-0436', 'scene-0437', 'scene-0438', 'scene-0439',
     'scene-0440', 'scene-0441', 'scene-0442', 'scene-0443', 'scene-0444', 'scene-0445', 'scene-0446', 'scene-0447',
     'scene-0448', 'scene-0449', 'scene-0450', 'scene-0451', 'scene-0452', 'scene-0453', 'scene-0454', 'scene-0455',
     'scene-0456', 'scene-0457', 'scene-0458', 'scene-0459', 'scene-0461', 'scene-0462', 'scene-0463', 'scene-0464',
     'scene-0465', 'scene-0467', 'scene-0468', 'scene-0469', 'scene-0471', 'scene-0472', 'scene-0474', 'scene-0475',
     'scene-0476', 'scene-0477', 'scene-0478', 'scene-0479', 'scene-0480', 'scene-0499', 'scene-0500', 'scene-0501',
     'scene-0502', 'scene-0504', 'scene-0505', 'scene-0506', 'scene-0507', 'scene-0508', 'scene-0509', 'scene-0510',
     'scene-0511', 'scene-0512', 'scene-0513', 'scene-0514', 'scene-0515', 'scene-0517', 'scene-0518', 'scene-0525',
     'scene-0526', 'scene-0527', 'scene-0528', 'scene-0529', 'scene-0530', 'scene-0531', 'scene-0532', 'scene-0533',
     'scene-0534', 'scene-0535', 'scene-0536', 'scene-0537', 'scene-0538', 'scene-0539', 'scene-0541', 'scene-0542',
     'scene-0543', 'scene-0544', 'scene-0545', 'scene-0546', 'scene-0566', 'scene-0568', 'scene-0570', 'scene-0571',
     'scene-0572', 'scene-0573', 'scene-0574', 'scene-0575', 'scene-0576', 'scene-0577', 'scene-0578', 'scene-0580',
     'scene-0582', 'scene-0583', 'scene-0584', 'scene-0585', 'scene-0586', 'scene-0587', 'scene-0588', 'scene-0589',
     'scene-0590', 'scene-0591', 'scene-0592', 'scene-0593', 'scene-0594', 'scene-0595', 'scene-0596', 'scene-0597',
     'scene-0598', 'scene-0599', 'scene-0600', 'scene-0639', 'scene-0640', 'scene-0641', 'scene-0642', 'scene-0643',
     'scene-0644', 'scene-0645', 'scene-0646', 'scene-0647', 'scene-0648', 'scene-0649', 'scene-0650', 'scene-0651',
     'scene-0652', 'scene-0653', 'scene-0654', 'scene-0655', 'scene-0656', 'scene-0657', 'scene-0658', 'scene-0659',
     'scene-0660', 'scene-0661', 'scene-0662', 'scene-0663', 'scene-0664', 'scene-0665', 'scene-0666', 'scene-0667',
     'scene-0668', 'scene-0669', 'scene-0670', 'scene-0671', 'scene-0672', 'scene-0673', 'scene-0674', 'scene-0675',
     'scene-0676', 'scene-0677', 'scene-0678', 'scene-0679', 'scene-0681', 'scene-0683', 'scene-0684', 'scene-0685',
     'scene-0686', 'scene-0687', 'scene-0688', 'scene-0689', 'scene-0695', 'scene-0696', 'scene-0697', 'scene-0698',
     'scene-0700', 'scene-0701', 'scene-0703', 'scene-0704', 'scene-0705', 'scene-0706', 'scene-0707', 'scene-0708',
     'scene-0709', 'scene-0710', 'scene-0711', 'scene-0712', 'scene-0713', 'scene-0714', 'scene-0715', 'scene-0716',
     'scene-0717', 'scene-0718', 'scene-0719', 'scene-0726', 'scene-0727', 'scene-0728', 'scene-0730', 'scene-0731',
     'scene-0733', 'scene-0734', 'scene-0735', 'scene-0736', 'scene-0737', 'scene-0738', 'scene-0739', 'scene-0740',
     'scene-0741', 'scene-0744', 'scene-0746', 'scene-0747', 'scene-0749', 'scene-0750', 'scene-0751', 'scene-0752',
     'scene-0757', 'scene-0758', 'scene-0759', 'scene-0760', 'scene-0761', 'scene-0762', 'scene-0763', 'scene-0764',
     'scene-0765', 'scene-0767', 'scene-0768', 'scene-0769', 'scene-0786', 'scene-0787', 'scene-0789', 'scene-0790',
     'scene-0791', 'scene-0792', 'scene-0803', 'scene-0804', 'scene-0805', 'scene-0806', 'scene-0808', 'scene-0809',
     'scene-0810', 'scene-0811', 'scene-0812', 'scene-0813', 'scene-0815', 'scene-0816', 'scene-0817', 'scene-0819',
     'scene-0820', 'scene-0821', 'scene-0822', 'scene-0847', 'scene-0848', 'scene-0849', 'scene-0850', 'scene-0851',
     'scene-0852', 'scene-0853', 'scene-0854', 'scene-0855', 'scene-0856', 'scene-0858', 'scene-0860', 'scene-0861',
     'scene-0862', 'scene-0863', 'scene-0864', 'scene-0865', 'scene-0866', 'scene-0868', 'scene-0869', 'scene-0870',
     'scene-0871', 'scene-0872', 'scene-0873', 'scene-0875', 'scene-0876', 'scene-0877', 'scene-0878', 'scene-0880',
     'scene-0882', 'scene-0883', 'scene-0884', 'scene-0885', 'scene-0886', 'scene-0887', 'scene-0888', 'scene-0889',
     'scene-0890', 'scene-0891', 'scene-0892', 'scene-0893', 'scene-0894', 'scene-0895', 'scene-0896', 'scene-0897',
     'scene-0898', 'scene-0899', 'scene-0900', 'scene-0901', 'scene-0902', 'scene-0903', 'scene-0945', 'scene-0947',
     'scene-0949', 'scene-0952', 'scene-0953', 'scene-0955', 'scene-0956', 'scene-0957', 'scene-0958', 'scene-0959',
     'scene-0960', 'scene-0961', 'scene-0975', 'scene-0976', 'scene-0977', 'scene-0978', 'scene-0979', 'scene-0980',
     'scene-0981', 'scene-0982', 'scene-0983', 'scene-0984', 'scene-0988', 'scene-0989', 'scene-0990', 'scene-0991',
     'scene-0992', 'scene-0994', 'scene-0995', 'scene-0996', 'scene-0997', 'scene-0998', 'scene-0999', 'scene-1000',
     'scene-1001', 'scene-1002', 'scene-1003', 'scene-1004', 'scene-1005', 'scene-1006', 'scene-1007', 'scene-1008',
     'scene-1009', 'scene-1010', 'scene-1011', 'scene-1012', 'scene-1013', 'scene-1014', 'scene-1015', 'scene-1016',
     'scene-1017', 'scene-1018', 'scene-1019', 'scene-1020', 'scene-1021', 'scene-1022', 'scene-1023', 'scene-1024',
     'scene-1025', 'scene-1044', 'scene-1045', 'scene-1046', 'scene-1047', 'scene-1048', 'scene-1049', 'scene-1050',
     'scene-1051', 'scene-1052', 'scene-1053', 'scene-1054', 'scene-1055', 'scene-1056', 'scene-1057', 'scene-1058',
     'scene-1074', 'scene-1075', 'scene-1076', 'scene-1077', 'scene-1078', 'scene-1079', 'scene-1080', 'scene-1081',
     'scene-1082', 'scene-1083', 'scene-1084', 'scene-1085', 'scene-1086', 'scene-1087', 'scene-1088', 'scene-1089',
     'scene-1090', 'scene-1091', 'scene-1092', 'scene-1093', 'scene-1094', 'scene-1095', 'scene-1096', 'scene-1097',
     'scene-1098', 'scene-1099', 'scene-1100', 'scene-1101', 'scene-1102', 'scene-1104', 'scene-1105', 'scene-1106',
     'scene-1107', 'scene-1108', 'scene-1109', 'scene-1110'],
'val':
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073'],
'mini_train':
    ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100'],
'mini_val':
    ['scene-0103', 'scene-0916'],    
}
    
if __name__ == '__main__':
  # with concurrent.futures.ProcessPoolExecutor() as executor:
  #     executor.map(main())
  #main()
  #executor = futures.ThreadPoolExecutor(max_workers=3)
  #executor.map(main())
  main()
    
    
    
    
    
    
    
    
    
    
    
    
