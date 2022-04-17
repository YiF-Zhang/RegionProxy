# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            '.ln': dict(decay_mult=0.),
            'dist_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)}))
optimizer_config = dict()
# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    periods=[10000, 20000, 50000],
    restart_weights=[1, 0.5, 0.25],
    min_lr_ratio=0,
    by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
