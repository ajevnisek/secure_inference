mkdir -p results/test_logs/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/after_finetune/
mkdir -p results/test_logs/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/before_finetune/
END=16
for RELU_NUM in $(seq 1 $END);
 do
   ./research/mmlab_tools/classification/dist_test_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_test_network_for_greedly_swaped_${RELU_NUM}_relu_after_finetuning.py 1 trained_networks/classification/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/latest.pth --metrics accuracy > results/test_logs/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/after_finetune/after_finetune_${RELU_NUM}.log
   ./research/mmlab_tools/classification/dist_test_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_finetune_network_for_greedly_swaped_${RELU_NUM}_relu.py 1 osi_checkpoints/cifar100/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/replaced_${RELU_NUM}_relus_greedy/greedy_search_for_${RELU_NUM}_induced_relus.pth --metrics accuracy > results/test_logs/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/before_finetune/before_finetune_${RELU_NUM}.log
done

