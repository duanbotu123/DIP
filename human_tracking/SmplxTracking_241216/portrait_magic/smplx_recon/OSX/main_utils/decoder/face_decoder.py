from ..config import cfg


channel_cfg = dict(
    num_output_channels=72,
    dataset_joints=72,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

emb_dim = 256
if cfg.upscale==1:
    neck_in_channels = [cfg.feat_dim]
    num_levels = 1
elif cfg.upscale==2:
    neck_in_channels = [cfg.feat_dim//2, cfg.feat_dim]
    # neck_in_channels = [768, 768]
    num_levels = 2
elif cfg.upscale==4:
    neck_in_channels = [cfg.feat_dim//4, cfg.feat_dim//2, cfg.feat_dim]
    # neck_in_channels = [768, 768, 768]
    num_levels = 3
elif cfg.upscale==8:
    neck_in_channels = [cfg.feat_dim//8, cfg.feat_dim//4, cfg.feat_dim//2, cfg.feat_dim]
    # neck_in_channels = [768, 768, 768, 768]
    num_levels = 4

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='Poseur',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', norm_cfg = norm_cfg, depth=50, num_stages=4, out_indices=(0, 1, 2, 3)),
    neck=dict(
        type='ChannelMapper',
        in_channels=neck_in_channels,
        kernel_size=1,
        out_channels=emb_dim,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
    ),
    keypoint_head=dict(
        type='Poseur_noise_sample',
        in_channels=512,
        num_queries=channel_cfg['num_output_channels'],
        num_reg_fcs=2,
        num_joints=channel_cfg['num_output_channels'],
        with_box_refine=True,
        loss_coord_enc=dict(type='RLELoss_poseur', use_target_weight=True),
        loss_coord_dec=dict(type='RLELoss_poseur', use_target_weight=True),
        # loss_coord_dec=dict(type='L1Loss', use_target_weight=True, loss_weight=5),
        loss_hp_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=10),
        # loss_coord_keypoint=dict(type='L1Loss', use_target_weight=True, loss_weight=1),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=emb_dim//2,
            normalize=True,
            offset=-0.5),
        transformer=dict(
            type='PoseurTransformer_v3',
            num_joints=channel_cfg['num_output_channels'],
            query_pose_emb = True,
            embed_dims = emb_dim,
            encoder=dict(
                type='DetrTransformerEncoder_zero_layer',
                num_layers=0,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    ffn_cfgs = dict(
                        embed_dims=emb_dim,
                        ),
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        num_levels=num_levels,
                        num_points=4,
                        embed_dims=emb_dim),
                    
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer_grouped',
                    ffn_cfgs = dict(
                        embed_dims=emb_dim,
                        ),
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=emb_dim,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention_post_value',
                            num_levels=num_levels,
                            num_points=4,
                            embed_dims=emb_dim)
                    ],
                    num_joints=channel_cfg['num_output_channels'],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        as_two_stage=True,
        use_heatmap_loss=False,
    ),
    train_cfg=dict(image_size=[192, 256]),
    test_cfg = dict(
        image_size=[192, 256],
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)
)

