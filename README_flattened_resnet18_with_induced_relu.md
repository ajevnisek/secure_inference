# Controlling Induced ReLUs in ResNet18

## What?

We created a script to control which layer induces which layer:
```shell
python flattened_resnets_controller.py
```

To validate that it yields a network which compiles, run:
```python
import torch
from flattened_resnet18_halfway_induced import HalfWayResNet18
choosing_matrices = torch.load('all_zeros_choosing_matrices_for_.HalfWayResNet18.pth')
model = HalfWayResNet18(choosing_matrices=choosing_matrices)
model(torch.rand(16, 3, 32, 32)).shape
```
# Why? 
Next, make mmcls know your network:
1) put the declaration in `mmpretrain/mmcls/models/backbones`,
2) change its header to include: 
```shell
from ..builder import BACKBONES

@BACKBONES.register_module()
```
3) and add it to the local `__init__.py`,

Now you can use it here:
```shell
./research/mmlab_tools/classification/dist_test_cls.sh  \
  research/configs/classification/flattened_resnet/baseline_resplit_cifar100_dataset_flattened_resnet18_halfway.py 1 \
  flattened_resnet18_checkpoint.pth   --metrics accuracy 

```