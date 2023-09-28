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
FCOS in MMDetection
Utilized existing FCOS configuration file, ???.
Conducted training for 10 epochs.
A learning rate of 0.0002 was used with a learning rate scheduler.
Executed the validation dataset after each epoch.

YOLOv8 in Ultralytics

![image](https://github.com/leehj825/cmpe249_hw1/assets/21224335/534fea02-31ce-410d-9e40-04259adbae8c)

<img width="400" alt="image" src="https://github.com/leehj825/cmpe249_hw1/assets/21224335/6d4c3f9f-3858-447d-a47d-0473a0fb2be1">


## Troubleshoot
TypeError: FormatCode() got an unexpected keyword argument 'verify'

for syntax problem: pip install yapf==0.40.1

