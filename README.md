# RadarNet Implementation

## Introduction

This repository implements a point-cloud-based object detection method called **RadarNet** (ECCV'20). 

**[Paper Link](https://arxiv.org/pdf/2007.14366.pdf)**

Please note:
- This repository reproduces only the object detection functionality from the original paper. The velocity estimation functionality will be added in future updates.
- Some specific details, such as learning rate, hard sample mining strategies, and sample extraction schemes, are not explicitly mentioned in the original paper. Hence, related works have been referenced to set up these parameters.

---

## Prerequisites

It is highly recommended to create a new environment using **Anaconda3** for this project.

Below is the version information for core libraries used:

| Library  | Version  |
| -------- | -------- |
| Python   | 3.7.10   |
| PyTorch  | 1.7.0    |
| CUDA     | 11.0     |

> **Note**:  
> The appropriate CUDA version depends on your GPU. Please ensure you install the correct version of CUDA and cuDNN on your server.  
> Additional libraries such as `nuscenes-devkit` and `opencv` may also be required when running the code. Install them as needed.

---

## Dataset Preparation

To train the network on the NuScenes dataset:
1. Download the dataset from [NuScenes](https://www.nuscenes.org/download).
2. Organize the dataset as follows:

```
NUSCENES_DATASET_ROOT/
├── samples/
├── sweeps/
├── maps/
├── v1.0-mini/
├── v1.0-trainval/
└── v1.0-test/
```

It is highly recommended to first test the code on the **mini** set before formally training the model on the **trainval** set.

After organizing the dataset, follow these steps:

1. Run the script at `src/tools/convert_nuScenes.py` to load and preprocess the required data.
   - Modify the dataset path and specify the splits you need in the script.
   - The script will extract the following:
     - Voxel representations of lidar and radar
     - Radar targets
     - Annotations
   - The processed data will be saved in the dataset root directory.

2. **Important Parameters**:
   - `NUM_SWEEPS_LIDAR` (Maximum: 10) and `NUM_SWEEPS_RADAR` (Maximum: 6) control the number of lidar and radar sweeps per sample.
   - To increase the influence of radar in detection, set `NUM_SWEEPS_LIDAR` to **1** (highly recommended).
   - In the original paper, these values were set to `10` and `6`, respectively, which significantly slows processing.

3. Ensure you have at least **300 GB** of disk space to store the fully prepared training data.

---

## Model Training

You can train the model using the `main.py` file. Follow these steps:

1. **Modify `num_chan`**:
   - This depends on the number of point cloud sweeps you use:
     ```
     num_chan = (NUM_SWEEPS_LIDAR * height_range / res_height) + NUM_SWEEPS_RADAR
     ```
   - `height_range` and `res_height` are defined in `convert_nuScenes.py`.

2. **Set Paths**:
   - Update `data_path` with the dataset directory.
   - Update `base_path` with the directory where you want to save training results.

3. **Resume Training**:
   - To resume training from a checkpoint, set `load_checkpoint = True`.

4. **Adjust Parameters**:
   - Variables like `batch_size`, `device`, learning rate, and its decay can be modified in `main.py`.
   - Epoch numbers, train splits, and other options can be configured in `opts.py`.

5. **Validation**:
   - To validate the model on the validation set after each epoch, uncomment the relevant lines at the bottom of the main function.

---

## Model Testing

1. Before testing, use `src/tools/visualize_result.py` to save BEV figures of lidar data for visualization.  
2. Run `test.py` to test the trained model on your dataset.  
3. Test results and visualizations will be saved in the directory specified by `res_path`.

---

## Data Analysis

1. Use `img2video.py` to convert visualization results into video format.  
2. Use `plot_loss.py` to generate training loss curves and AP (Average Precision) curves.

---

## Notes

This repository provides a rough implementation of the overall project. If you wish to extend its functionality, you are encouraged to implement additional features as needed.
