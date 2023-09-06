_base_ = [
    '../_base_/datasets/cifar100_resplit_bs64.py',
    '../_base_/schedules/cifar10_bs128.py',
]
def load_choosing_matrices():
    import pickle
    with open("temp_dir/replace_ten_and_then_the_rest/associations/layer_name_to_choosing_matrix/layer_name_to_matrix_merged.pkl", 'rb') as f:
        choosing_matrices = pickle.load(f)
    return choosing_matrices


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetInduceInParts_1',
        choosing_matrices=load_choosing_matrices(),

    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.001)
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)

# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='gloo')
log_level = 'INFO'
load_from = None
resume_from = 'temp_dir/replace_ten_and_then_the_rest/checkpoints/checkpoint_1.pth'

evaluation = dict(interval=10, by_epoch=True)
workflow = [('train', 10), ('val', 1), ('val', 1),]

