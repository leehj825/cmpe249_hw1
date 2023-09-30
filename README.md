# CMPE 249 Homework #1

## Table of Contents

- [Introduction](https://github.com/leehj825/cmpe249_hw1/edit/main/README.md#introduction)
- [Models and Code Bases](https://github.com/leehj825/cmpe249_hw1/edit/main/README.md#models-and-code-bases)
- [Dataset Preparation](https://github.com/leehj825/cmpe249_hw1/edit/main/README.md#dataset-preparation)
- Training
    - FCOS in MMDetection
    - YOLOv8 in Ultralytics
- Result
- Troubleshoot
- Reference

## Introduction
This homework is to explore some training deep learning models for 2D object detection. Two models from different code bases, FCOS from MMDetection and YOLOv8 from Ultralytics, have been chosen for comparison in performance.

## Models and Code Bases
- **Model 1**: FCOS (Fully Convolutional One-Stage Object Detection)
    - Code Base: MMDetection (https://github.com/open-mmlab/mmdetection)
- **Model 2**: YOLOv8
    - Code Base: Ultralytics (https://github.com/ultralytics/yolov8)

## Dataset Preparation
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

Based on an existing FCOS config file (fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_coco.py), I changed the dataset path to 1000 samples of Waymo data converted to COCO format (~/coco_1k) wth annotation files.  Based on the new dataset, the data pipelines and loaders are updated.  
```
data_root = '../../coco_1k/'  # Root path of data
# Path of train annotation file
train_ann_file = 'annotations/result.json'
train_data_prefix = 'images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/result.json'
val_data_prefix = 'images/'  # Prefix of val image path
```
The training script is provided under tools folder.
```
> python tools/train.py fcos_x101-64x4d_fpn_gn-head-1x_waymococo_hj.py
```
The number of epochs are reduced to 10 for shorter training time. The new classes are defined based on the Waymo dataset. 
```
max_epochs = 10  # Maximum training epochs
num_classes = 4  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 2
val_batch_size_per_gpu = 1

save_epoch_intervals = 1
max_keep_ckpts = 3

classes = ('vehicle','pedestrian', 'sign', 'cyclist')
```
Below training process shows the progress of the last epoch #10.  The total training time including the validations between each epoch was about 2.75 hours.
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

Ultralytics documentation provides the example python script to train, validate, and run the model with minimal modification of the config files. The line for loading a pretrained model is commented out to start the training from scratch.  The epoch is set to 10 same as the FCOS model above.

```
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
#model = YOLO('yolov8n.pt')

# Train the model using the 'waymo_coco.yaml' dataset for 10 epochs
results = model.train(data='waymo_coco.yaml', epochs=10)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')
```
Below is the list of parameters and layers used for YOLOv8 algorithm. 
```
                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
:                          
:          
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    752092  ultralytics.nn.modules.head.Detect           [4, [64, 128, 256]]           
YOLOv8n summary: 225 layers, 3011628 parameters, 3011612 gradients
```
The training time is about **0.2 hour** much shorter than FCOS model.
```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/10     0.692G     0.7905     0.6145     0.9118         60        640: 100%|██████████| 250/250
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|████████
                   all       1000       7887      0.932      0.812      0.863      0.615

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/10     0.673G     0.7793     0.5869     0.9004         36        640: 100%|██████████| 250/250
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|████████
                   all       1000       7887      0.924      0.814      0.867      0.649

10 epochs completed in 0.198 hours.
Optimizer stripped from /home/001891254/cmpe249_hw1/ultralytics/runs/detect/train3/weights/last.pt, 6.2MB
Optimizer stripped from /home/001891254/cmpe249_hw1/ultralytics/runs/detect/train3/weights/best.pt, 6.2MB
```
## Result
The trainings from the scratch with 10 epochs, FCOS model shows better metric than YOLOv8.  However considering the trainig time, YOLOv8 is more than 10 times faster than FCOS. 
Model | Training time (hours)
--- | ---
FCOS | 2.75
YOLOv8 | 0.198 
<img width="541" alt="image" src="https://github.com/leehj825/cmpe249_hw1/assets/21224335/732b18fc-9552-433d-850d-13dff3f38d55">

### Inference
Over-detection is found in both models, by putting multiple bounding boxes around each object. 
#### FCOS

<img width="541" alt="image" src="https://github.com/leehj825/cmpe249_hw1/assets/21224335/8a306dc8-6272-48f0-8bd9-3633c744b141">

#### YOLOv8
<img width="541" alt="image" src="https://github.com/leehj825/cmpe249_hw1/assets/21224335/61f3a8b3-17f8-45ad-a763-e02fc15b9727">



## Troubleshoot
TypeError: FormatCode() got an unexpected keyword argument 'verify'

for syntax problem: pip install yapf==0.40.1

