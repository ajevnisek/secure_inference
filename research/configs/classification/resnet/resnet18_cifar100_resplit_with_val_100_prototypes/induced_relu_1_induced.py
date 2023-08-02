import pickle
import os.path as osp

NUM_OF_REPLACEMENTS = 1


_base_ = [
    '../../_base_/datasets/cifar100_bs64.py',
    '../../_base_/schedules/cifar10_bs128.py',
]

removed_sequence = [ 
    'layer4[1].relu_2', 
    'layer1[0].relu_1', 
    'layer4[1].relu_1', 
    'layer2[0].relu_2', 
    'layer1[1].relu_1', 
    'layer1[1].relu_2', 
    'layer1[0].relu_2', 
    'layer4[0].relu_2', 
    'layer2[1].relu_2', 
    'layer2[1].relu_1', 
    'layer3[0].relu_1', 
    'layer3[0].relu_2', 
    'layer2[0].relu_1', 
    'layer3[1].relu_1', 
    'layer3[1].relu_2', 
    'layer4[0].relu_1', 
] 
def load_resnet18_choosing_matrices():
    with open(
        osp.join(
            'cached_relus/cifar100_with_val/resnet18/100_most_important_relus/layer_name_to_choosing_matrix/layer_name_to_matrix.pkl'
        ),
        "rb",
    ) as f:
        layer_name_to_matrix = pickle.load(f)
    return layer_name_to_matrix
def layer_name_to_row_col(layer_name):
    row = int(layer_name.split("layer")[-1].split("[")[0]) - 1
    level = int(layer_name.split("[")[-1].split("]")[0])
    sub = int(layer_name.split("_")[-1])
    col = 2 * int(level) + sub - 1
    return row, col


USE_INDUCED_RELU = [
    [False] * 2 * 2
    for layer in [
        2,
    ]
    * 4
]
for r in removed_sequence[:NUM_OF_REPLACEMENTS]:
    row, col = layer_name_to_row_col(r)
    USE_INDUCED_RELU[row][col] = True


def load_permutation_matrices_for_resnet18():
    layer_name_to_matrix = load_resnet18_choosing_matrices()
    permutation_matrices = []
    for layer_index in [1, 2, 3, 4]:
        for sub_index in [0, 1]:
            for act_index in [1, 2]:
                layer_name = f"layer{layer_index}[{sub_index}].relu_{act_index}"
                permutation_matrices.append(layer_name_to_matrix[layer_name].cuda())
    return permutation_matrices


# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        # type='ResNet_CIFAR',
        # depth=18,
        # num_stages=4,
        # out_indices=(3, ),
        # style='pytorch',
        type="ResNet_InducedReLU",
        num_classes=100,
        is_cifar=True,
        permutation_matrices=load_permutation_matrices_for_resnet18(),
        use_induced_relu=USE_INDUCED_RELU,
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=100,
        in_channels=512,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)

optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.001)
lr_config = dict(policy="step", step=[60, 120, 160], gamma=0.2)

# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable

dist_params = dict(backend="gloo")
log_level = "INFO"
load_from = None
resume_from = None
evaluation = dict(interval=10, by_epoch=True)
workflow = [("train", 10), ("val", 1)]

