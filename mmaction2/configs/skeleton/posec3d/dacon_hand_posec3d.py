gpu_ids = ['cuda:0']

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=21,  # joint
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=5,  # 1
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=dict(),
    test_cfg=dict(average_clips='score')
)

dataset_type = 'PoseDataset'
ann_file_train = '/home/sckim/Dataset/Competition/dacon_hand/train.pkl'
ann_file_test = '/home/sckim/Dataset/Competition/dacon_hand/test.pkl'

skeletons = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (12, 0),
             (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=30),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

val_pipeline = [
    # dict(
    #     type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    # dict(type='PoseDecode'),
    # dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    # dict(type='Resize', scale=(-1, 64)),
    # dict(type='CenterCrop', crop_size=56),
    # dict(
    #     type='GeneratePoseTarget',
    #     sigma=0.6,
    #     use_score=True,
    #     with_kp=True,
    #     with_limb=False,
    #     # double=True,
    #     # left_kp=left_kp,
    #     # right_kp=right_kp,
    #     skeletons=skeletons),
    # dict(type='FormatShape', input_format='NCTHW'),
    # dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    # dict(type='ToTensor', keys=['imgs'])
]

# val_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1920, 1280),
#         # scale_factor = test_mt_ratios,
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(
#                 type='Normalize',
#                 mean=[123.675, 116.28, 103.53],
#                 std=[58.395, 57.12, 57.375],
#                 to_rgb=True),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]


test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=30, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=56),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=16,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix='',
        class_prob={1},
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix='',
        pipeline=test_pipeline),
)

# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0003)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)

total_epochs = 50
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(
    interval=1,
    metrics=['accuracy', 'precision', 'recall', 'f1_score'])

log_config = dict(
    interval=1, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/sckim/Dataset/Competition/dacon_hand/'
load_from = None
resume_from = None
find_unused_parameters = False
