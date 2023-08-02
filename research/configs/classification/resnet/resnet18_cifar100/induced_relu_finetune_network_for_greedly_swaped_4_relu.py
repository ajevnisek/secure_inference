import pickle
import os.path as osp


_base_ = [
    '../../_base_/datasets/cifar100_bs64.py',
    '../../_base_/schedules/cifar10_bs128.py',
]

USE_INDUCED_RELU = [[False] * 2 * 2 for layer in [2,] * 4]
USE_INDUCED_RELU[0][0] = True
USE_INDUCED_RELU[3][-1] = True
USE_INDUCED_RELU[0][2] = True
USE_INDUCED_RELU[0][1] = True

DEFAULT_PICLE_PATH = osp.join('choosing_matrices', 'cifar100', 'resnet18',
                              'layer_name_to_choosing_matrix',
                              f'resnet18_layer_name_to_matrix.pkl')


def load_resnet18_choosing_matrices(layer_name_to_matrix_pickle_path=DEFAULT_PICLE_PATH):
    with open(layer_name_to_matrix_pickle_path, 'rb') as f:
        layer_name_to_matrix = pickle.load(f)
    return layer_name_to_matrix


def load_permutation_matrices_for_resnet18():
    layer_name_to_matrix = load_resnet18_choosing_matrices()
    permutation_matrices = []
    for layer_index in [1, 2, 3, 4]:
        for sub_index in [0, 1]:
            for act_index in [1, 2]:
                layer_name = f'layer{layer_index}[{sub_index}].act{act_index}'
                permutation_matrices.append(layer_name_to_matrix[layer_name])
    return permutation_matrices


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        # type='ResNet_CIFAR',
        # depth=18,
        # num_stages=4,
        # out_indices=(3, ),
        # style='pytorch',
        type='ResNet_InducedReLU',
        num_classes=100,
        is_cifar=True,
        permutation_matrices=load_permutation_matrices_for_resnet18(),
        use_induced_relu=USE_INDUCED_RELU
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
resume_from = 'osi_checkpoints/cifar100/resnet18/replaced_4_relus_greedy/greedy_search_for_four_induced_relus.pth'
evaluation = dict(interval=10, by_epoch=True)
workflow = [('train', 10), ('val', 1)]