import os
import subprocess


DEFAULT_CONFIGS_DIR = 'research/configs/classification/flattened_resnet/'
DEFAULT_CHECKPOINT_PATH = '/mnt/data/secure_inference_cache/trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/baseline/latest.pth'
DEFAULT_ACTIVATIONS_DIR = 'cached_relus/cifar100_with_val/flattened_resnet18'
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_ACTIVATIONS_DIR, 'activations.pkl')


def create_cache_config_file(model_name: str, output_path: str,
                             is_induced_network: bool = False,
                             choosing_matrices_full_path: str = ''):
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
    '../_base_/datasets/cifar100_resplit_bs64.py',
    '../_base_/schedules/cifar10_bs128.py',
]
""" + slice1 + f"""

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='{model_name}',
        """ + slice2 + """

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
resume_from = None
evaluation = dict(interval=10, by_epoch=True)
workflow = [('train', 10), ('val', 1), ('val', 1),]"""
    with open(output_path, 'w') as f:
        f.write(config)



def cache_activations(model_name: str = 'FlattenedResNet18',
                      configs_dir: str = DEFAULT_CONFIGS_DIR,
                      checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
                      activations_cache_dir: str = DEFAULT_ACTIVATIONS_DIR,
                      activations_output_path: str = DEFAULT_OUTPUT_PATH,
                      is_induced_network: bool = False,
                      choosing_matrices_full_path: str = ''):
    config_path = os.path.join(configs_dir, f'{model_name}_activations_cache_config.py')
    create_cache_config_file(model_name, config_path, is_induced_network, choosing_matrices_full_path)
    # Run the other script
    #import pdb
    #pdb.set_trace()
    subprocess.run(["python3", "mmpretrain/tools/test_with_hooks_flattened_networks.py",
                    config_path, checkpoint_path, '--out', f'{activations_cache_dir}/out_file_flattened.json',
                    '--metrics', 'accuracy',
                    '--activations_out', f'{activations_output_path}'])


if __name__ == '__main__':
    cache_activations(model_name='FlattenedResNet18',
                      configs_dir='research/configs/classification/flattened_resnet_split_to_two',
                      checkpoint_path='/mnt/data/secure_inference_cache/trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/baseline/latest.pth',
                      activations_cache_dir='temp_dir/replace_ten_and_then_the_rest/activations',
                      activations_output_path='temp_dir/replace_ten_and_then_the_rest/activations/activations_0.pkl',
                      is_induced_network=False)
    cache_activations('ResNet18OneLayerWithSelfInduceToTenLayers',
                      'research/configs/classification/flattened_resnet_split_to_two',
                      'trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/replacing_first_ten_layerswith_self_induce/latest.pth',
                      'temp_dir/replace_ten_and_then_the_rest/activations',
                      'temp_dir/replace_ten_and_then_the_rest/activations/activations_1.pkl',
                      is_induced_network=True,
                      choosing_matrices_full_path='cached_relus/cifar100_with_val/flattened_resnet18/layer_name_to_choosing_matrix/layer_name_to_matrix.pkl')
