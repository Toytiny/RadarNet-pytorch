## Data preparation 

You can use the convert.nuScenes.py file in the src/tools to aggregate and prepare all needed data from nuScenes dataset and save them to .json files. The data in this these files can be used in the training or inference stage. Please modify the DATA_PATH as the file path of your downloaded nuscenes dataset at first. Note that in the DATA_PATH, the files should be in the format as follows (taking the v1.0-mini as an example):

--DATA_PATH

​    --sample

​    --sweeps

​    --maps

​    --v1.0-mini

## JSON files

1. After aggregation, each .json file in the the voxel_representation folder includes a) the extracted voxel representations of the sensor data, b) the projected radar target information in the BEV view, for a single sample. These files can be used to train the network or to test the model. 
2. After aggregation, each .json file in the annotations folder includes the groundtruth of detection bounding boxes within the RoI (Region of Interest) in the BEV view. Note that each .json file in this folder corresponds to a file in the voxel_representation folder. These files are used only in the training process.

## Parameters

You can modify the RoI by changing the values of side_range, fwd_range, and height range. All the data has been tranformed to the ego-vehicle's centric coordinate using the sensor and ego pose from the dataset.

You can also change the resolution of the voxels by changing the values of res_height and res_wl. 

1. NUM_SWEEPS_LIDAR denotes the number of used lidar sweeps for one sample.  The lidar used to collect the nuscenes dataset is 20Hz. So, to aggregate the data in the past 0.5s, the value of NUM_SWEEPS_LIDAR should not be bigger than 10.
2. NUM_SWEEPS_RADAR denotes the number of used radar sweeps for one sample.  The radar used to collect the nuscenes dataset is 13Hz. So, to aggregate the data in the past 0.5s, the value of NUM_SWEEPS_LIDAR should not be bigger than 6.

