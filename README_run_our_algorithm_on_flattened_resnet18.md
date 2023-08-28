1) Use a baseline flattened resnet network:

 
a) create a checkpoint using the previous unflattened network checkpoint:
```python
from flatten_networks import convert_old_format_checkpoint_to_new_format_checkpoint
path_to_old_resnet18_checkpoint = 'trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth'
path_to_new_resnet18_checkpoint = 'trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/baseline/latest.pth'
convert_old_format_checkpoint_to_new_format_checkpoint(path_to_old_resnet18_checkpoint, path_to_new_resnet18_checkpoint)
```
b) Validate that the accuracy remains as before:
run the previous baseline:
```shell
./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/baseline_resplit_cifar100_dataset.py 1   trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth   --metrics accuracy
```
run the current baseline:
```shell
CONFIG=research/configs/classification/flattened_resnet/baseline_resplit_cifar100_dataset_flattened_resnet18.py
CHECKPOINT=trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/baseline/latest.pth
./research/mmlab_tools/classification/dist_test_cls.sh ${CONFIG} 1 ${CHECKPOINT} --metrics accuracy
```
2) create a network with your configuration of induced / inducer layers:
```python
from flattened_resnets_controller import ResNetController
    from flattened_resnet18 import FlattenedResNet18
flattened_model = FlattenedResNet18()
controller = ResNetController(new_name='ResNet18SwapFirstLayerWithInducedReLU', 
                              new_path='flattened_resnet18_swap_first_layer_with_induced_relu.py',
                              old_model=flattened_model,
                              inducer_to_induced={
                                  'relu0': ['ResLayer0_BasicBlockV20_relu_1']})
controller.create_declaration()
controller.add_declaration_to_mmcls_backbones()
```
3) Cache ReLU activations:
```shell
CONFIG=research/configs/classification/flattened_resnet/baseline_resplit_cifar100_dataset_flattened_resnet18.py
CHECKPOINT=trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/baseline/latest.pth
ACTS_CACHE=cached_relus/cifar100_with_val/flattened_resnet18/activations.pkl
mkdir -p cached_relus/cifar100_with_val/flattened_resnet18/10_most_important_relus
python mmpretrain/tools/test_with_hooks_flattened_networks.py ${CONFIG} ${CHECKPOINT} --out out_file_flattened.json --metrics accuracy --activations_out ${ACTS_CACHE} 
```
4) Cluster ReLU activations:
a) create an inducer to induced json:
```python
import json
d = {'relu0': ['ResLayer0_BasicBlockV20_relu_1']}
with open('flattened_resnet18_one_induced_layer.json', 'w') as f:
    json.dump(d, f, indent=4)
```
 b) run clusterer:
```shell
INDUCER_TO_INDUCED='flattened_resnet18_one_induced_layer.json'
NUM_PROTOTYPES=10
ACTS_CACHE=cached_relus/cifar100_with_val/flattened_resnet18/activations.pkl
OUT_DIR=cached_relus/cifar100_with_val/flattened_resnet18/${NUM_PROTOTYPES}_most_important_relus
mkdir -p cached_relus/cifar100_with_val/flattened_resnet18/${NUM_PROTOTYPES}_most_important_relus

python flattened_networks_cluster_script.py ${INDUCER_TO_INDUCED} ${NUM_PROTOTYPES} ${ACTS_CACHE} ${OUT_DIR} 

```
5) create permutation matrices:
```shell
mkdir -p results/cifar100_with_val/flattened_resnet18/${NUM_PROTOTYPES}_most_important_prototypes/
python create_choosing_matrices_fixed.py ${NUM_PROTOTYPES} cached_relus/cifar100_with_val/resnet18/

```