import torch

from mmcv import Config
from mmcls.models import build_classifier
from mmcls.datasets import build_dataloader


activation = {}
def get_drelu_values(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = (output.detach() != input[0].detach()).float()
  return hook



model_cfg = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR_V2',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
model = build_classifier(model_cfg)



cfg = Config.fromfile('research/configs/classification/resnet/resnet18_cifar100/baseline.py')
from mmcls.datasets import build_dataset
datasets = [build_dataset(cfg.data.train)]
loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=False,
        round_up=True,
        seed=cfg.get('seed'),
        sampler_cfg=cfg.get('sampler', None),
    )
loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}


if 'distortion_extraction' in cfg.data:
    del cfg.data['distortion_extraction']
loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=False,
        round_up=True,
        seed=cfg.get('seed'),
        sampler_cfg=cfg.get('sampler', None),
    )
    # The overall dataloader settings
loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}
train_loader_cfg['workers_per_gpu'] = 1
data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in datasets[0]]

model.backbone.relu.register_forward_hook(get_drelu_values('firstReLU'))
model.backbone.layer4[0].relu.register_forward_hook(get_drelu_values(
    'layer4_0_relu'))
result = model.backbone(torch.rand(16, 3, 32, 32))

