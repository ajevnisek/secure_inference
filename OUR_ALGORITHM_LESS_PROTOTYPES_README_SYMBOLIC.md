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
mkdir -p cached_relus/cifar100_with_val/resnet18/100_most_important_relus
python mmpretrain/tools/test_with_hooks.py trained_networks/classification/resnet18_cifar100_resplit/baseline/baseline_resplit_cifar100_dataset.py trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth --out out_file.json --metrics accuracy --activations_out cached_relus/cifar100_with_val/resnet18/activations.pkl
```
---

3. Cluster ReLUs with Farthest Point Sampling: 
```shell
NUM_PROTOTYPES=500
mkdir -p cached_relus/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_relus
#python find_correlated_relus.py cached_relus/cifar100_with_val/resnet18/activations.pkl cached_relus/cifar100_with_val/resnet18/
python find_correlated_relus_with_less_prototypes.py \
  cached_relus/cifar100_with_val/resnet18/activations.pkl \
  ${NUM_PROTOTYPES} cached_relus/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_relus
```

4. Create the (permutation) choosing matrices:
```shell
mkdir -p results/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_prototypes/

cp cached_relus/cifar100_with_val/resnet18/activations.pkl \
 cached_relus/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_relus/
cp cached_relus/cifar100_with_val/resnet18/1K_prototypes_* \
 cached_relus/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_relus/ 
python create_choosing_matrices.py \
 cached_relus/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_relus
```
5. Run the greedy search algorithm:
```shell
mkdir -p research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/
cp  research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/baseline_resplit_cifar100_dataset.py research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/
./research/mmlab_tools/classification/dist_greedy_test_cls.sh  \ 
 research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/baseline_resplit_cifar100_dataset.py \ 
 1   trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth \ 
 cached_relus/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_relus/layer_name_to_choosing_matrix/layer_name_to_matrix.pkl \
   results/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_prototypes/ \
    --metrics accuracy
```
6. Fix the config files according to the grid-search results:
```shell
python get_greedily_removed_order.py results/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_prototypes cached_relus/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_relus/layer_name_to_choosing_matrix/layer_name_to_matrix.pkl research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes osi_checkpoints/cifar100/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes trained_networks/classification/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes
```
7.Test the New ResNet18 with Induced ReLUs using:
```shell
./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_0_induced.py 1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth --metrics accuracy
./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_1_induced.py 1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth --metrics accuracy
./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_4_induced.py 1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth --metrics accuracy
./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_16_induced.py 1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth --metrics accuracy

 ```
8. Finetune the networks with:
```shell
END=16
for RELU_NUM in $(seq 1 $END);
 do echo $RELU_NUM;
 ./research/mmlab_tools/classification/dist_create_induced_relu_cached_model.sh  research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_finetune_network_for_greedly_swaped_${RELU_NUM}_relu.py  1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth  ${RELU_NUM} osi_checkpoints/cifar100/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes  --removed_sequence research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/removed_sequence.pkl; ./research/mmlab_tools/classification/dist_test_cls.sh  research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_${RELU_NUM}_induced.py  1 osi_checkpoints/cifar100/resnet18/replaced_${RELU_NUM}_relus_greedy/greedy_search_for_${RELU_NUM}_induced_relus.pth  --metrics accuracy ; ./research/mmlab_tools/classification/dist_finetune_train_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_finetune_network_for_greedly_swaped_${RELU_NUM}_relu.py 1   --work-dir trained_networks/classification/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/ ; ./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_test_network_for_greedly_swaped_${RELU_NUM}_relu_after_finetuning.py 1 trained_networks/classification/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/latest.pth --metrics accuracy
done
```
9. Test the networks with:
```shell
mkdir -p results/test_logs/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/after_finetune/
mkdir -p results/test_logs/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/before_finetune/
END=16
for RELU_NUM in $(seq 1 $END);
 do
   ./research/mmlab_tools/classification/dist_test_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_test_network_for_greedly_swaped_${RELU_NUM}_relu_after_finetuning.py 1 trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/latest.pth --metrics accuracy > results/test_logs/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/after_finetune/after_finetune_${RELU_NUM}.log
   ./research/mmlab_tools/classification/dist_test_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_finetune_network_for_greedly_swaped_${RELU_NUM}_relu.py 1 osi_checkpoints/cifar100/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/replaced_${RELU_NUM}_relus_greedy/greedy_search_for_${RELU_NUM}_induced_relus.pth --metrics accuracy > results/test_logs/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/before_finetune/before_finetune_${RELU_NUM}.log
done
```
10. Plot the graphs with:
```shell
python plot_finetuning_result_cifar100_resnet18.py ${NUM_PROTOTYPES}
```

---
