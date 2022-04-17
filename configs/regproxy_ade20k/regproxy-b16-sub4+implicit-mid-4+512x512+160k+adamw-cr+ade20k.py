_base_ = [
    '../_base_/models/regproxy/regproxy-b16.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/adamw+cr+lr_6e-5+wd_0.01+iter_160k.py'
]
model = dict(
    backbone=dict(
        deit=True,
        out_indices=[4, 11]),
    decode_head=dict(
        num_classes=150))