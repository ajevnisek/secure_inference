# Our algorithm for reducing relu count


1. Train a network with a train, validation and test sets:

```shell
BASELINE_DIR=trained_networks/classification/resnet18_cifar100_resplit/baseline/
./research/mmlab_tools/classification/dist_train_cls.sh \
  research/configs/classification/resnet/resnet18_cifar100/baseline_resplit_cifar100_dataset.py 1 \
  --work-dir ${BASELINE_DIR}
```
2. Test it with:
```shell
./research/mmlab_tools/classification/dist_test_cls.sh \
  research/configs/classification/resnet/resnet18_cifar100/baseline_resplit_cifar100_dataset.py 1 \
  trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth \
  --metrics accuracy  
  
```
2. Cache ReLU activations of the validation set with:
```shell
mkdir -p cached_relus/cifar100_with_val/resnet18/
python mmpretrain/tools/test_with_hooks.py trained_networks/classification/resnet18_cifar100_resplit/baseline/baseline_resplit_cifar100_dataset.py trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth --out out_file.json --metrics accuracy --activations_out cached_relus/cifar100_with_val/resnet18/activations.pkl
```

3. Cluster ReLUs with Farthest Point Sampling: 
```shell
python find_correlated_relus.py cached_relus/cifar100_with_val/resnet18/activations.pkl cached_relus/cifar100_with_val/resnet18/
```

4. Create the (permutation) choosing matrices:
```shell
python create_choosing_matrices.py cached_relus/cifar100_with_val/resnet18/
```

5. Test the New ResNet18 with Induced ReLUs using:
```shell
./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_01_induced.py 1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth --metrics accuracy
./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_04_induced.py 1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth --metrics accuracy
./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_16_induced.py 1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth --metrics accuracy

 ```
6. Finetune the Induced ReLU backbone with:
```shell
./research/mmlab_tools/classification/dist_train_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_finetune_network_for_greedly_swaped_4_relu.py 1   --work-dir trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_4_relus_finetune/

```
7. Test the New ResNet18 network with:
```shell
./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_test_network_for_greedly_swaped_4_relu_after_finetuning.py 1 trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_4_relus_finetune/latest.pth --metrics accuracy
```
8. TBD



## For 6 Induced ReLUs:
1. Create the Induced ReLU cached model:
```shell
./research/mmlab_tools/classification/dist_create_induced_relu_cached_model.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_finetune_network_for_greedly_swaped_6_relu.py 1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth 6
```
2. Evaluate result model to prove accuracy drop:
```shell
./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_06_induced.py 1 osi_checkpoints/cifar100/resnet18/replaced_6_relus_greedy/greedy_search_for_six_induced_relus.pth --metrics accuracy
```
3. Finetune:
```shell
./research/mmlab_tools/classification/dist_finetune_train_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_finetune_network_for_greedly_swaped_6_relu.py 1   --work-dir trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_6_relus_finetune/
```
4. Evaluate fine-tuned model to observe accuracy:
```shell
./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_test_network_for_greedly_swaped_6_relu_after_finetuning.py 1 trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_6_relus_finetune/latest.pth --metrics accuracy
```
## FOR RELU_NUM:
```shell
RELU_NUM=7; ./research/mmlab_tools/classification/dist_create_induced_relu_cached_model.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_finetune_network_for_greedly_swaped_${RELU_NUM}_relu.py 1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth ${RELU_NUM} ; ./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_${RELU_NUM}_induced.py 1 osi_checkpoints/cifar100/resnet18/replaced_${RELU_NUM}_relus_greedy/greedy_search_for_${RELU_NUM}_induced_relus.pth --metrics accuracy ; ./research/mmlab_tools/classification/dist_finetune_train_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_finetune_network_for_greedly_swaped_${RELU_NUM}_relu.py 1   --work-dir trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/ ;./research/mmlab_tools/classification/dist_test_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_test_network_for_greedly_swaped_${RELU_NUM}_relu_after_finetuning.py 1 trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/latest.pth --metrics accuracy
RELU_NUM=12; ./research/mmlab_tools/classification/dist_create_induced_relu_cached_model.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_finetune_network_for_greedly_swaped_${RELU_NUM}_relu.py 1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth ${RELU_NUM} ; ./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_${RELU_NUM}_induced.py 1 osi_checkpoints/cifar100/resnet18/replaced_${RELU_NUM}_relus_greedy/greedy_search_for_${RELU_NUM}_induced_relus.pth --metrics accuracy ; ./research/mmlab_tools/classification/dist_finetune_train_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_finetune_network_for_greedly_swaped_${RELU_NUM}_relu.py 1   --work-dir trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/ ;./research/mmlab_tools/classification/dist_test_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_test_network_for_greedly_swaped_${RELU_NUM}_relu_after_finetuning.py 1 trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/latest.pth --metrics accuracy

```
