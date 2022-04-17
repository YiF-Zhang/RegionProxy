_base_ = [
    '../_base_/models/regproxy/regproxy-l16.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/adamw+cr+lr_6e-5+wd_0.01+iter_80k.py'
]
model = dict(
    backbone=dict(
        img_size=(768, 768),
        out_indices=[5, 23]),
    test_cfg=dict(
        mode='slide',
        crop_size=(768, 768),
        stride=(512, 512)))
