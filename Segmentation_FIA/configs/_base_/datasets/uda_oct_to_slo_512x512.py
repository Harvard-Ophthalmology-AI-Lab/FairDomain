# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'OctSloFundsDataset'
data_root = '/scratch/cw3437/Data/fairdomain/'
img_norm_cfg = dict(
    mean=[165.8527, 165.8527, 165.8527], std=[27.2024, 27.2024, 27.2024], to_rgb=True)
crop_size = (512, 512)

oct_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
slo_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img','gt_semantic_seg']),
            dict(type='Collect', keys=['img','gt_semantic_seg','attr_label']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='OctSloFundsDataset',
            data_root='/scratch/cw3437/Data/fairdomain/',
            img_dir='images',
            ann_dir='labels',
            list_dir = "lists/FairDomain_final/",
            attr_label="race", 
            split="train", 
            fundus_split = "fundus_oct",
            pipeline=oct_train_pipeline,
            ),
        target=dict(
            type='OctSloFundsDataset',
            data_root='/scratch/cw3437/Data/fairdomain/',
            img_dir='images',
            ann_dir='labels',
            list_dir = "lists/FairDomain_final/",
            attr_label="race", 
            split="train", 
            fundus_split = "fundus_slo",
            pipeline=slo_train_pipeline,
            )),
    # val=dict(
    #     type='OctSloFundsDataset',
    #     data_root='/scratch/cw3437/Data/fairdomain/',
    #     img_dir='leftImg8bit/val',
    #     ann_dir='gtFine/val',
    #     pipeline=test_pipeline),
    test=dict(
        type='OctSloFundsDataset',
        data_root='/scratch/cw3437/Data/fairdomain/',
        img_dir='images',
        ann_dir='labels',
        list_dir = "lists/FairDomain_final/",
        attr_label="race", 
        split="test", 
        fundus_split = "fundus_slo",
        pipeline=test_pipeline,))
