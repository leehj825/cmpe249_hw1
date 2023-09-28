# CMPE 249 Homework #1

## Training option:

**Option1:** You need to have at least two models from different code base, e.g., different models in YOLOv5/v8 (https://github.com/ultralytics) is considered as one code base. YOLOv7 (https://github.com/WongKinYiu/yolov7), mmyolo (https://github.com/open-mmlab/mmyolo Links to an external site.), mmdetection (https://github.com/open-mmlab/mmdetectionLinks to an external site.), detectron2 (https://github.com/facebookresearch/detectron2) are different code bases.

**Selection:** MMDetection with FCOS and Ultralytics with YOLOv8

## Dataset preparation
Waymo dataset in SJSU HPC server (/data/cmpe249-fa23/waymotrain200cocoyolo) with 161,096 image files are copied to a different location and reduced to 1000 images to shorten the training time. The dataset format is initially in YOLO format with label files.

- `images`: contains image files
- `labels`: contains label files for **YOLO** format
- `annotations`: contains annotation json files for **COCO** format.

To convert YOLO format to COCO format:
```
python yolo2coco.py .
```

## MMDetection
https://github.com/open-mmlab/mmdetection

## Ultralytics
https://github.com/ultralytics

![image](https://github.com/leehj825/cmpe249_hw1/assets/21224335/534fea02-31ce-410d-9e40-04259adbae8c)

<img width="400" alt="image" src="https://github.com/leehj825/cmpe249_hw1/assets/21224335/6d4c3f9f-3858-447d-a47d-0473a0fb2be1">


## Troubleshoot
TypeError: FormatCode() got an unexpected keyword argument 'verify'

for syntax problem: pip install yapf==0.40.1

