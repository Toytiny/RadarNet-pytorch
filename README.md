# RadarNet Implementation



## Introduction



This repository implemented a point-cloud-based object detection method called RadarNet (ECCV'20). 

Paper Link:  https://arxiv.org/pdf/2007.14366.pdf

Please note that this repository only reproduce the object detection function in the original paper, and the  velocity estimation function will be added in the future. BTW, some specific details, such as the learning rate, hard sample mining strategy, sample extraction scheme, and so on, are not well-illustrated in the original paper, so here I refer to related works for setting up. 



## Prerequisite



I highly recommenced the user to create a new environment using Anaconda3. 

The version of some core libraries I used can be seen in the table below:

| library | version |
| ------- | ------- |
| Python  | 3.7.10  |
| Pytorch | 1.7.0   |
| CUDA    | 11.0    |

 Please note that, the CUDA version depends on your GPU version, please install proper CUDA and cudnn on your server. Other libraries, like nuscenes-devkit and opencv, maybe also needed when running the code, please install them if necessary.



## Dataset Preparation



To train the network on the nuscenes datasets, you should download the data on https://www.nuscenes.org/download. You should organized the data like:

```
NUSCENES_DATASET_ROOT/
├── samples/
├── sweeps/
├── maps/
├── v1.0-mini/
├── v1.0-trainval/
└── v1.0-test/
```

I highly recommended you to first test the code on the mini set and formally train the model on the trainval set.

After organizing the dataset, you should first run the .src/tools/convert_nuScenes.py to load and process the data you need. You must modify the data path, and choose the splits you need in the file. The file extracts the voxel representations of lidar and radar, the radar target, as well as the annotations, and save them is the dataset root path. 

Please note that, the max value of NUM_SWEEPS_LIDAR is 10, while the max value of NUM_SWEEPS_RADAR is 6. These two variables control the number of sweeps of lidar (radar) in a sample. If you want to improve the weight of radar in the detection, you may set the NUM_SWEEPS_LIDAR to 1 (highly-recommended). In the experiments of the original paper, they set 10 and 6 for NUM_SWEEPS_LIDAR and NUM_SWEEPS_RADAR, however, it has a very slow speed. Note that, you may need over 300G space to store all prepared data of the training set. 



## Model training



You can train the model with the main.py file. You should first modify the variable num_chan according to how many sweeps of point cloud you use. For example, if your NUM_SWEEPS_LIDAR is 3, and NUM_SWEEPS_RADAR is 6, the num_chan=3*height_range/res_height+6. (height_range and res_height are two values in convert_nuScenes.py)

Then you should modify the data_path and the base_path which is the place you save the training result. If you want to resume the training and load the checkpoint to continue the training, please set the load_checkpoint to True. Some variables, like the batch_size, device, learning rate and its decay can be modified in this file. Other variables, like the epoch number, train_split, used in the file can be set in the opts.py file. 



If you want to validate the model on the validation set after each epoch, please delete the comments in the bottom of the main function.





## Model Test



Before model test, you should use the ./src/tools/visualize_result.py to the save the bev figures of the lidar data for visualiazation. Then you could run the test.py to test the model you trained on the dataset you need, the visualization of the test results in save in the "res_path".



## Data Analysis



You could use the img2video.py to convert the visualization results to video. You could also use the plot loss to get the curve of the training loss and AP curve. 



## Note



This repository is a rough version for the overall project. If you want to add other functions to it, you can write the code by yourself. 
