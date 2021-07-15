

import os
import json
import ujson
import numpy as np

import copy
import time
import sys

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    
import matplotlib.pyplot as plt
import time
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

LIDAR_LIST=['LIDAR_TOP']
DATA_PATH = '/home/toytiny/Desktop/RadarNet/data/nuscenes/'
SPLITS = {
          'mini_val': 'v1.0-mini',
          #'mini_train': 'v1.0-mini',
          #'train': 'v1.0-trainval',
          #'val': 'v1.0-trainval',
          #'test': 'v1.0-test',
          }
SCENE_SPLITS = {
'mini_train':
    ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100'],
'mini_val':
    ['scene-0103', 'scene-0916'],    
}
    
NUM_SWEEPS_LIDAR=3

OUT_PATH='/home/toytiny/Desktop/RadarNet2/figures/'


def scale_to_255(a, min, max, dtype=np.uint8):
	return ((a - min) / float(max - min) * 255).astype(dtype)
 
 


def bev_generator(points):
    
    side_range = (-50, 50)  
    fwd_range = (-50, 50)
    res = 0.15625
    height_range = (-3,5)
    x_points = points[0,:]
    y_points = points[1,:]
    z_points = points[2,:]
 
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    h_filt = np.logical_and((z_points > height_range[0]), (z_points < height_range[1]))
    filt = np.logical_and(np.logical_and(f_filt, s_filt),h_filt)
        
    indices = np.argwhere(filt).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
   
    # SHIFT to the BEV view
    x_img = x_points + fwd_range[1] 
    y_img = -(y_points + side_range[0])
    z_img = z_points
    
    
    x_img = (x_img / res).astype(np.int32)
    y_img = (y_img / res).astype(np.int32)
    
    
    #print(x_img.min(), x_img.max(), y_img.min(), x_img.max())
     
    
    pixel_value = scale_to_255(z_points, height_range[0], height_range[1])
    x_max = int((fwd_range[1] - fwd_range[0]) / res)
    y_max = int((side_range[1] - side_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    
    return im 
    
 

def nuscenes_lidar_bev():
    
    for split in SPLITS:
        
        out_path = OUT_PATH + split 
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        data_path = DATA_PATH
        nusc = NuScenes(
            version=SPLITS[split], dataroot=data_path, verbose=True)
        num_scenes = 0
        num_pcs = 0
        
        for sample in nusc.sample:
            scene_name = nusc.get('scene', sample['scene_token'])['name']
            if not (split in ['test']) and not (scene_name in SCENE_SPLITS[split]):
                continue
            if sample['prev'] == '':
                print('scene_name', scene_name)
                num_scenes+= 1
                # skip the first keyframe since it has no prev sweeps  
                continue
      
            # Load lidar points from files and transform them to car coordinate  
            print('generating lidar data BEV images for pcs:', num_pcs)
            pc_token = sample['data'][LIDAR_LIST[0]]
            pc_data = nusc.get('sample_data', pc_token)
            num_pcs += 1
            if num_pcs<0:
                out_path_current=out_path+'/'+'bev_scenes-{}_pcs-0{}.jpg'.format(num_scenes,num_pcs)
            else:
                out_path_current=out_path+'/'+'bev_scenes-{}_pcs-{}.jpg'.format(num_scenes,num_pcs)
            if os.path.exists(out_path_current):
                continue
            # Complex coordinate transform from Lidar to car
            sd_record = nusc.get('sample_data', pc_token)
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
            trans_matr = transform_matrix(
                     cs_record['translation'], Quaternion(cs_record['rotation']),
                     inverse=False)
            
            lidar_pcs, _ = LidarPointCloud.from_file_multisweep(nusc, 
                               sample, LIDAR_LIST[0], LIDAR_LIST[0], NUM_SWEEPS_LIDAR)
            
            for i in range(0, len(lidar_pcs)):
                lidar_pcs[i][:3, :] = trans_matr.dot(np.vstack((lidar_pcs[i][:3, :], 
                                   np.ones(lidar_pcs[i].shape[1]))))[:3, :]
                if i==0:
                    all_lidar=lidar_pcs[0]
                else:
                    all_lidar=np.hstack((all_lidar,lidar_pcs[i]))
      
        
            
            img=bev_generator(all_lidar)
            #cv2.imshow('Image',img)
            #cv2.waitKey(0)
            cv2.imwrite(out_path_current, img)   #save picture
            #print(1)
            
            
            
        
      











if __name__ == '__main__':
  nuscenes_lidar_bev()    