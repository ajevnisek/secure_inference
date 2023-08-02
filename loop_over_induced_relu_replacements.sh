#END=16
#for RELU_NUM in $(seq 1 $END);
# do echo $RELU_NUM;
# ./research/mmlab_tools/classification/dist_create_induced_relu_cached_model.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_finetune_network_for_greedly_swaped_${RELU_NUM}_relu.py 1 trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth ${RELU_NUM} ; ./research/mmlab_tools/classification/dist_test_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_${RELU_NUM}_induced.py 1 osi_checkpoints/cifar100/resnet18/replaced_${RELU_NUM}_relus_greedy/greedy_search_for_${RELU_NUM}_induced_relus.pth --metrics accuracy ; ./research/mmlab_tools/classification/dist_finetune_train_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_finetune_network_for_greedly_swaped_${RELU_NUM}_relu.py 1   --work-dir trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/ ;./research/mmlab_tools/classification/dist_test_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_test_network_for_greedly_swaped_${RELU_NUM}_relu_after_finetuning.py 1 trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/latest.pth --metrics accuracy
#done
#END=16
#for RELU_NUM in $(seq 1 $END);
# do
#   ./research/mmlab_tools/classification/dist_test_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_test_network_for_greedly_swaped_${RELU_NUM}_relu_after_finetuning.py 1 trained_networks/classification/resnet18_cifar100_resplit/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/latest.pth --metrics accuracy > results/test_logs/after_finetune/after_finetune_${RELU_NUM}.log
#   ./research/mmlab_tools/classification/dist_test_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/induced_relu_finetune_network_for_greedly_swaped_${RELU_NUM}_relu.py 1 osi_checkpoints/cifar100/resnet18/replaced_6_relus_greedy/greedy_search_for_six_induced_relus.pth --metrics accuracy > results/test_logs/before_finetune/before_finetune_${RELU_NUM}.log
#done

