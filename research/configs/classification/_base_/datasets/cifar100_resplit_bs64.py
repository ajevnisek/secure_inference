# dataset settings
dataset_type = 'CIFAR100WithVal'
img_norm_cfg = dict(
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

distortion_extraction_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/cifar100-new-split',
        pipeline=train_pipeline, test_mode='train'),
    val=dict(
        type=dataset_type,
        data_prefix='data/cifar100-new-split',
        pipeline=val_pipeline,
        test_mode='val'),
    test=dict(
        type=dataset_type,
        data_prefix='data/cifar100-new-split',
        pipeline=test_pipeline,
        test_mode='test'),
    distortion_extraction=dict(
        type=dataset_type,
        data_prefix='data/cifar100-new-split',
        pipeline=distortion_extraction_pipeline),
)