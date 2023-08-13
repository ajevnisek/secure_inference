mkdir -p trained_networks/${NUM_PROTOTYPES}_finetune_attempt
./research/mmlab_tools/classification/dist_finetune_train_cls.sh   research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/swap_16_more_epochs.py 1 --work-dir trained_networks/${NUM_PROTOTYPES}_finetune_attempt/
./research/mmlab_tools/classification/dist_test_cls.sh research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_test_network_for_greedly_swaped_${RELU_NUM}_relu_after_finetuning.py 1 trained_networks/classification/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes/induced_relu_backbone_greedly_swap_${RELU_NUM}_relus_finetune/latest.pth --metrics accuracy

