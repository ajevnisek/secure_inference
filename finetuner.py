import os
import subprocess


DEFAULT_CONFIGS_DIR = 'research/configs/classification/flattened_resnet/'
DEFAULT_CHECKPOINT_PATH = '/mnt/data/secure_inference_cache/trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/baseline/latest.pth'
DEFAULT_ACTIVATIONS_DIR = 'cached_relus/cifar100_with_val/flattened_resnet18'
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_ACTIVATIONS_DIR, 'activations.pkl')


def create_finetune_config_file(model_name: str,
                                output_path: str,
                                num_epochs : int,
                                is_induced_network: bool = False,
                                choosing_matrices_full_path: str = '',
                                previous_checkpoint: str = '',
                                use_entire_train_set=False):
    
    if not use_entire_train_set:
        dataset_path = '../_base_/datasets/cifar100_resplit_bs64.py'
    else:
        dataset_path = '../_base_/datasets/cifar100_bs64.py'
        
    if not is_induced_network:
        slice1 = ''
        slice2 = ''
    else:
        slice1 = f"""def load_choosing_matrices():
        
    import pickle
    with open("{choosing_matrices_full_path}", 'rb') as f:
        choosing_matrices = pickle.load(f)
    return choosing_matrices
"""
        slice2 = "choosing_matrices=load_choosing_matrices(),"

    config = f"""_base_ = [
    '{dataset_path}',
    '../_base_/schedules/cifar10_bs128.py',
]
""" + slice1 + f"""

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='{model_name}',
        """ + slice2 + f"""

    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

runner = dict(type='EpochBasedRunner', max_epochs={180 + num_epochs})
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.001)
lr_config = dict(policy='step', step=[60, 120, 160, 250, 320], gamma=0.2)

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
load_from = None"""
    if previous_checkpoint:
        config += f"""
resume_from = '{previous_checkpoint}'
"""
    else:
        config += f"""
resume_from = None
"""
    config += """
evaluation = dict(interval=10, by_epoch=True)
workflow = [('train', 10), ('val', 1), ('val', 1),]

"""
    with open(output_path, 'w') as f:
        f.write(config)


def run_finetuning(finetune_config_file, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    subprocess.call(['./research/mmlab_tools/classification/dist_finetune_train_cls.sh', finetune_config_file, '1',
                     '--work-dir', results_dir])
