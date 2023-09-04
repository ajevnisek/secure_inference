import os
import argparse

def parse_args():

	parser = argparse.ArgumentParser(description='create mmcls configs for induced models')

	parser.add_argument('base_config_path', type=str, help='the base mmcls config for the induced model')
	parser.add_argument('finetune_config_path', type=str, help='the config used to finetune the model')
	parser.add_argument('model_name', type=str, help='the induced model name that is registered by mmcls')
	parser.add_argument('permutation_matrices_path', type=str, help='the path to the matrix mapping')
	parser.add_argument('base_induced_checkpoint', type=str, help='the intitial checkpoint path for the induced model')

	return parser.parse_args() 

def create_base_config(args):

	model_name = args.model_name
	permutation_matrices_path = args.permutation_matrices_path
	base_config_output_path = args.base_config_path

	base_config_content = f"""

import os
import pickle

_base_ = [
   	'../_base_/datasets/cifar100_resplit_bs64.py',
    '../_base_/schedules/cifar10_bs128.py',
]
def load_choosing_matrices():
    with open('{os.path.abspath(permutation_matrices_path)}', 'rb') as f:
        choosing_matrices = pickle.load(f)
    return choosing_matrices

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='{model_name}',
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
resume_from = None
evaluation = dict(interval=10, by_epoch=True)
workflow = [('train', 10), ('val', 1), ('val', 1),]
"""
	
	with open(base_config_output_path, 'w') as f:
		f.write(base_config_content)

def create_finetune_config(args):
	model_name = args.model_name
	permutation_matrices_path = args.permutation_matrices_path
	finetune_config_output_path = args.finetune_config_path
	base_induced_checkpoint = args.base_induced_checkpoint

	finetune_config_content = f"""

import os
import pickle 

_base_ = [
   	'../_base_/datasets/cifar100_resplit_bs64.py',
    '../_base_/schedules/cifar10_bs128.py',
]
def load_choosing_matrices():
    with open('{os.path.abspath(permutation_matrices_path)}', 'rb') as f:
        choosing_matrices = pickle.load(f)
    return choosing_matrices

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='{model_name}',
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
resume_from = '{os.path.abspath(base_induced_checkpoint)}'
evaluation = dict(interval=10, by_epoch=True)
workflow = [('train', 10), ('val', 1), ('val', 1),]
"""

	with open(finetune_config_output_path, 'w') as f:
		f.write(finetune_config_content)
def main():

	args = parse_args()

	create_base_config(args)
	create_finetune_config(args)

if __name__ == "__main__":
	main()