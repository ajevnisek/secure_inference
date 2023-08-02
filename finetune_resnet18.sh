BASELINE_DIR=trained_networks/classification/resnet18_cifar100/baseline/
INDUCED_RELUS_BACKBONE_ALL_RELUS=trained_networks/classification/resnet18_cifar100/induced_relu_backbone_all_relus/
INDUCED_RELUS_BACKBONE_SWAP_FOUR_GREEDILY=trained_networks/classification/resnet18_cifar100/induced_relu_backbone_greedly_swap_4_relus_retrain/
INDUCED_RELUS_BACKBONE_SWAP_FOUR_GREEDILY_FINETUNE=trained_networks/classification/resnet18_cifar100/induced_relu_backbone_greedly_swap_4_relus_finetune/


./research/mmlab_tools/classification/dist_train_cls.sh \
  research/configs/classification/resnet/resnet18_cifar100/induced_relu_finetune_network_for_greedly_swaped_4_relu.py 1 \
  --work-dir ${INDUCED_RELUS_BACKBONE_SWAP_FOUR_GREEDILY_FINETUNE}


./research/mmlab_tools/classification/dist_train_cls.sh \
  research/configs/classification/resnet/resnet18_cifar100/induced_relu_retrain_network_for_greedly_swaped_4_relu.py 1 \
  --work-dir ${INDUCED_RELUS_BACKBONE_SWAP_FOUR_GREEDILY}


./research/mmlab_tools/classification/dist_train_cls.sh \
   research/configs/classification/resnet/resnet18_cifar100/induced_relu_baseline_all_relus.py 1 \
   --work-dir ${INDUCED_RELUS_BACKBONE_ALL_RELUS}


./research/mmlab_tools/classification/dist_train_cls.sh \
  research/configs/classification/resnet/resnet18_cifar100/baseline.py 1 \
  --work-dir ${BASELINE_DIR}
