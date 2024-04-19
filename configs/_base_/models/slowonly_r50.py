model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained='torchvision://resnet50',
        lateral=False,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False),
    # neck=dict(
    #     type='TPN',
    #     in_channels=(256, 512),
    #     out_channels=256,
    #     spatial_modulation_cfg=dict(
    #         in_channels=(256, 512), out_channels=512),
    #     temporal_modulation_cfg=dict(downsample_scales=(4, 8)),
    #     upsample_cfg=dict(scale_factor=(2, 1, 1)),
    #     downsample_cfg=dict(downsample_scale=(2, 1, 1)),
    #     level_fusion_cfg=dict(
    #         in_channels=(256, 256),
    #         mid_channels=(256, 256),
    #         out_channels=512,
    #         downsample_scales=((2, 1, 1), (1, 1, 1))),
    #     aux_head_cfg=dict(out_channels=48, loss_weight=0.5)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=48,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(fcn_test=True))
