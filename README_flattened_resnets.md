# Flattened ResNet model

## Goal
Our goal is to create a "flattened" version of a ResNet model. That is, get rid of the classes implementation of the resnet and just use the resnet primitives: 
```shell
Conv2D
ReLU
BatchNorm2d
```
Why?

Since we want now to have a very simple and intuitive control over the definitions of:
1) for which layers we calculate ReLUs
2) for which layers we use InducedReLUs

## Approach
We wrote a script:
```shell
python flatten_networks.py
```
which does the following:
1) it creates a new file called `flatten_resnet18.py` which is a flattened model declaration of the ResNet18 model.
2) it augments a checkpoint such that it can load a checkpoint of the original ResNet18 model into the flattened version.

Our approach for this script was to iterate over the model's children (sub modules) and:
1) push them down a queue which holds them as elements we initialize in the class's init.
2) create a forward pass queue which holds the computational graph in order. 
3) track the original children names and the new flattened objects names such that we can augment the original ResNet state dict.

## Usage
1) Run the following python:
```shell
python flatten_networks.py
```

2) Copy `flatten_resnet18.py` to `mmpretrain/mmcls/models/backbones` and change its header to:
```python 
import torch
import torch.nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, Identity, Sequential
from ..builder import BACKBONES

@BACKBONES.register_module()
class FlattenedResNet18(torch.nn.Module):
```
such that the model gets registered.

Optionally validate features dimensions consistency with:
```shell
python flatten_networks_run.py
```

3) Create a new configuration script for the new backbone:
```shell
research/configs/classification/flattened_resnet/baseline_resplit_cifar100_dataset_flattened_resnet18.py
```
4) Test it with:
```shell
./research/mmlab_tools/classification/dist_test_cls.sh \
  research/configs/classification/flattened_resnet/baseline_resplit_cifar100_dataset_flattened_resnet18.py 1  \
  flattened_resnet18_checkpoint.pth   --metrics accuracy 
```