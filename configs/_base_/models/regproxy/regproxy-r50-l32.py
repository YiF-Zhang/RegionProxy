norm_cfg = dict(type='SyncBN', requires_grad=True)
global_channels = 1024
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='VisionTransformerHybrid',
        embed_type='vit_large_r50_s32_384',
        img_size=(512, 512),
        patch_size=32,
        in_channels=3,
        embed_dims=global_channels,
        num_layers=24,
        num_heads=global_channels // 64,
        mlp_ratio=4,
        out_indices=(-1, 23),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=True,
        final_norm=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic'),
    decode_head=dict(
        type='ProxyHead',
        in_channels=(global_channels, global_channels),
        in_index=(-2, -1),
        input_transform='multiple_select',
        channels=global_channels,
        dropout_ratio=0.0,
        num_classes=19,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        region_res=(4, 4)),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512)))
