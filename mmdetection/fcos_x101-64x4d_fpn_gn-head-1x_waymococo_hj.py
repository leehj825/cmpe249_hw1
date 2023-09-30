_base_ = 'configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py'
dataset_type = 'CocoDataset'
# ========================Frequently modified parameters======================
# -----data related-----
data_root = '../../coco_1k/'  # Root path of data
# Path of train annotation file
train_ann_file = 'annotations/result.json'
train_data_prefix = 'images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/result.json'
val_data_prefix = 'images/'  # Prefix of val image path

max_epochs = 10  # Maximum training epochs
num_classes = 4  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 2
val_batch_size_per_gpu = 1

save_epoch_intervals = 1
max_keep_ckpts = 3

classes = ('vehicle','pedestrian', 'sign', 'cyclist')
#load_from = '~/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'
# model settings
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_cfg = dict(max_epochs=max_epochs)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=_base_.backend_args))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        metainfo=dict(classes=classes),
        data_prefix=dict(img=val_data_prefix),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=_base_.backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox',
    format_only=False,
    backend_args=_base_.backend_args)
test_evaluator = val_evaluator


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[4,8],
        gamma=0.1)
]
