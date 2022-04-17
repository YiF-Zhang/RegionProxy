_base_ = [
    '../_base_/models/regproxy/regproxy-r50-l32.py',
    '../_base_/datasets/ade20k_640x640.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/adamw+cr+lr_6e-5+wd_0.01+iter_160k.py'
]
model = dict(
    backbone=dict(
        img_size=(640, 640),
        out_indices=[-1, 23]),
    decode_head=dict(
        num_classes=150),
    test_cfg=dict(
        mode='slide',
        crop_size=(640, 640),
        stride=(640, 640)))