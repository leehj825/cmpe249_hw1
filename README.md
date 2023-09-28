# CMPE 249 Homework #1

## Introduction
This homework is to explore some training deep learning models for 2D object detection. Two models from different code bases, FCOS from MMDetection and YOLOv8 from Ultralytics, have been chosen for comparison in performance.

## Models and Code Bases
- **Model 1**: FCOS (Fully Convolutional One-Stage Object Detection)
    - Code Base: MMDetection (https://github.com/open-mmlab/mmdetection)
- **Model 2**: YOLOv8
    - Code Base: Ultralytics (https://github.com/ultralytics/yolov8)

## Dataset preparation
Waymo dataset in SJSU HPC server (/data/cmpe249-fa23/waymotrain200cocoyolo) with 161,096 image files are copied to a different location and reduced to 1000 images to shorten the training time. The dataset format is initially in YOLO format with label files.

- `images`: contains image files
- `labels`: contains label files for **YOLO** format
- `annotations`: contains annotation json files for **COCO** format.

To convert YOLO format to COCO format:
```
python yolo2coco.py .
```
## Training
**FCOS in MMDetection**

Existing FCOS configuration file, ???.
Conducted training for 10 epochs.
A learning rate of 0.0002 was used with a learning rate scheduler.
Executed the validation dataset after each epoch.

```
2023/09/27 01:33:11 - mmengine - INFO - Epoch(train) [10][ 50/500]  base_lr: 1.0000e-04 lr: 1.0000e-04  eta: 0:11:48  time: 1.5860  data_time: 0.0074  memory: 9232  grad_norm: 2.3275  loss: 0.8868  loss_cls: 0.0740  loss_bbox: 0.2408  loss_centerness: 0.5720
2023/09/27 01:34:30 - mmengine - INFO - Epoch(train) [10][100/500]  base_lr: 1.0000e-04 lr: 1.0000e-04  eta: 0:10:29  time: 1.5701  data_time: 0.0052  memory: 9232  grad_norm: 2.3272  loss: 0.8759  loss_cls: 0.0651  loss_bbox: 0.2352  loss_centerness: 0.5756
2023/09/27 01:35:48 - mmengine - INFO - Epoch(train) [10][150/500]  base_lr: 1.0000e-04 lr: 1.0000e-04  eta: 0:09:11  time: 1.5672  data_time: 0.0049  memory: 9232  grad_norm: 2.5081  loss: 0.8404  loss_cls: 0.0642  loss_bbox: 0.2045  loss_centerness: 0.5717
2023/09/27 01:37:08 - mmengine - INFO - Epoch(train) [10][200/500]  base_lr: 1.0000e-04 lr: 1.0000e-04  eta: 0:07:52  time: 1.6053  data_time: 0.0049  memory: 9232  grad_norm: 2.7784  loss: 0.8986  loss_cls: 0.0843  loss_bbox: 0.2438  loss_centerness: 0.5705
2023/09/27 01:38:27 - mmengine - INFO - Epoch(train) [10][250/500]  base_lr: 1.0000e-04 lr: 1.0000e-04  eta: 0:06:33  time: 1.5728  data_time: 0.0049  memory: 9232  grad_norm: 2.6551  loss: 0.8699  loss_cls: 0.0696  loss_bbox: 0.2294  loss_centerness: 0.5709
2023/09/27 01:39:46 - mmengine - INFO - Epoch(train) [10][300/500]  base_lr: 1.0000e-04 lr: 1.0000e-04  eta: 0:05:14  time: 1.5742  data_time: 0.0050  memory: 9232  grad_norm: 2.9299  loss: 0.8669  loss_cls: 0.0790  loss_bbox: 0.2168  loss_centerness: 0.5711
2023/09/27 01:41:05 - mmengine - INFO - Epoch(train) [10][350/500]  base_lr: 1.0000e-04 lr: 1.0000e-04  eta: 0:03:56  time: 1.5779  data_time: 0.0049  memory: 9232  grad_norm: 2.5208  loss: 0.8745  loss_cls: 0.0750  loss_bbox: 0.2280  loss_centerness: 0.5715
2023/09/27 01:42:23 - mmengine - INFO - Epoch(train) [10][400/500]  base_lr: 1.0000e-04 lr: 1.0000e-04  eta: 0:02:37  time: 1.5725  data_time: 0.0048  memory: 9232  grad_norm: 2.7218  loss: 0.8775  loss_cls: 0.0751  loss_bbox: 0.2282  loss_centerness: 0.5743
2023/09/27 01:43:42 - mmengine - INFO - Epoch(train) [10][450/500]  base_lr: 1.0000e-04 lr: 1.0000e-04  eta: 0:01:18  time: 1.5727  data_time: 0.0049  memory: 9232  grad_norm: 2.4860  loss: 0.8744  loss_cls: 0.0717  loss_bbox: 0.2290  loss_centerness: 0.5737
2023/09/27 01:45:02 - mmengine - INFO - Exp name: fcos_x101-64x4d_fpn_gn-head-1x_waymococo_hj_20230926_224612
2023/09/27 01:45:02 - mmengine - INFO - Epoch(train) [10][500/500]  base_lr: 1.0000e-04 lr: 1.0000e-04  eta: 0:00:00  time: 1.5954  data_time: 0.0047  memory: 9232  grad_norm: 2.8320  loss: 0.8441  loss_cls: 0.0723  loss_bbox: 0.2026  loss_centerness: 0.5692
2023/09/27 01:45:02 - mmengine - INFO - Saving checkpoint at 10 epochs
```

**YOLOv8 in Ultralytics**

```
                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    752092  ultralytics.nn.modules.head.Detect           [4, [64, 128, 256]]           
YOLOv8n summary: 225 layers, 3011628 parameters, 3011612 gradients
```

![image](https://github.com/leehj825/cmpe249_hw1/assets/21224335/534fea02-31ce-410d-9e40-04259adbae8c)

<img width="400" alt="image" src="https://github.com/leehj825/cmpe249_hw1/assets/21224335/6d4c3f9f-3858-447d-a47d-0473a0fb2be1">


## Troubleshoot
TypeError: FormatCode() got an unexpected keyword argument 'verify'

for syntax problem: pip install yapf==0.40.1

