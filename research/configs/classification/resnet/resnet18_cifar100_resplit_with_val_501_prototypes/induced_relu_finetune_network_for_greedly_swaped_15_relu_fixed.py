dataset_type = 'CIFAR100'
img_norm_cfg = dict(
    mean=[129.304, 124.07, 112.434], std=[68.17, 65.392, 70.418], to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[129.304, 124.07, 112.434],
        std=[68.17, 65.392, 70.418],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
distortion_extraction_pipeline = [
    dict(
        type='Normalize',
        mean=[129.304, 124.07, 112.434],
        std=[68.17, 65.392, 70.418],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(
        type='Normalize',
        mean=[129.304, 124.07, 112.434],
        std=[68.17, 65.392, 70.418],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='CIFAR100',
        data_prefix='data/cifar100',
        pipeline=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CIFAR100',
        data_prefix='data/cifar100',
        pipeline=[
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CIFAR100',
        data_prefix='data/cifar100',
        pipeline=[
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    distortion_extraction=dict(
        type='CIFAR100',
        data_prefix='data/cifar100',
        pipeline=[
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]))
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
runner = dict(type='EpochBasedRunner', max_epochs=200)
NUM_OF_REPLACEMENTS = 15
removed_sequence = [
    'layer4[1].relu_2', 'layer1[0].relu_1', 'layer1[1].relu_2',
    'layer2[0].relu_2', 'layer1[1].relu_1', 'layer1[0].relu_2',
    'layer2[1].relu_1', 'layer3[1].relu_1', 'layer3[1].relu_2',
    'layer2[1].relu_2', 'layer3[0].relu_1', 'layer3[0].relu_2',
    'layer4[0].relu_1', 'layer2[0].relu_1', 'layer4[0].relu_2',
    'layer4[1].relu_1'
]
DEFAULT_PICLE_PATH = 'cached_relus/cifar100_with_val/resnet18/501_most_important_relus/layer_name_to_choosing_matrix/layer_name_to_matrix.pkl'
USE_INDUCED_RELU = [[True, True, True, True], [True, True, True, True],
                    [True, True, True, True], [True, True, False, True]]
r = 'layer4[0].relu_2'
row = 3
col = 1
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_InducedReLU',
        num_classes=100,
        is_cifar=True,
        permutation_matrices=[
            tensor(
                indices=tensor(
                    [[35252, 35252, 35252, ..., 35252, 35252, 35252],
                     [0, 1, 2, ..., 65533, 65534, 65535]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 65536),
                nnz=65536,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[31746, 38323, 32162, ..., 30719, 30719, 30719],
                     [0, 1, 2, ..., 65533, 65534, 65535]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 65536),
                nnz=65536,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[35252, 62572, 62572, ..., 35252, 35252, 35252],
                     [0, 1, 2, ..., 65533, 65534, 65535]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 65536),
                nnz=65536,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[62656, 32003, 32162, ..., 18425, 18425, 18425],
                     [0, 1, 2, ..., 65533, 65534, 65535]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 65536),
                nnz=65536,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[50208, 30815, 30815, ..., 35252, 62572, 12254],
                     [0, 1, 2, ..., 32765, 32766, 32767]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 32768),
                nnz=32768,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[31821, 31821, 31821, ..., 55257, 55257, 32735],
                     [0, 1, 2, ..., 32765, 32766, 32767]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 32768),
                nnz=32768,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[57568, 35252, 35252, ..., 35252, 35252, 35252],
                     [0, 1, 2, ..., 32765, 32766, 32767]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 32768),
                nnz=32768,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[31821, 31821, 31821, ..., 18296, 32735, 32735],
                     [0, 1, 2, ..., 32765, 32766, 32767]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 32768),
                nnz=32768,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[35252, 28191, 28191, ..., 63486, 63486, 28632],
                     [0, 1, 2, ..., 16381, 16382, 16383]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 16384),
                nnz=16384,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[35252, 35252, 35252, ..., 35252, 35252, 35252],
                     [0, 1, 2, ..., 16381, 16382, 16383]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 16384),
                nnz=16384,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[35252, 35252, 35252, ..., 32567, 32664, 38323],
                     [0, 1, 2, ..., 16381, 16382, 16383]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 16384),
                nnz=16384,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[57610, 28191, 28191, ..., 35252, 35252, 35523],
                     [0, 1, 2, ..., 16381, 16382, 16383]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 16384),
                nnz=16384,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[27801, 62572, 62572, ..., 63486, 63486, 52186],
                     [0, 1, 2, ..., 8189, 8190, 8191]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 8192),
                nnz=8192,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[35252, 35252, 35252, ..., 35252, 35252, 35252],
                     [0, 1, 2, ..., 8189, 8190, 8191]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 8192),
                nnz=8192,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor(
                    [[35252, 35252, 35252, ..., 35252, 35252, 35252],
                     [0, 1, 2, ..., 8189, 8190, 8191]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 8192),
                nnz=8192,
                layout=torch.sparse_coo),
            tensor(
                indices=tensor([[19456, 19456, 19456, ..., 7061, 7061, 50908],
                                [0, 1, 2, ..., 8189, 8190, 8191]]),
                values=tensor([1., 1., 1., ..., 1., 1., 1.]),
                size=(65536, 8192),
                nnz=8192,
                layout=torch.sparse_coo)
        ],
        use_induced_relu=[[True, True, True, True], [True, True, True, True],
                          [True, True, True, True], [True, True, False,
                                                     True]]),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
checkpoint_config = dict(interval=10)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='gloo')
log_level = 'INFO'
load_from = None
resume_from = 'osi_checkpoints/cifar100/resnet18_cifar100_resplit_with_val_501_prototypes/replaced_15_relus_greedy/greedy_search_for_15_induced_relus.pth'
evaluation = dict(interval=10, by_epoch=True)
workflow = [('train', 10), ('val', 1), ('val', 1)]
work_dir = 'trained_networks/classification/resnet18_cifar100_resplit_with_val_501_prototypes/induced_relu_backbone_greedly_swap_15_relus_finetune/'
gpu_ids = range(0, 1)
