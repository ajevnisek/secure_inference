BASELINE_DIR=trained_networks/classification/resnet18_cifar100_resplit/baseline/

./research/mmlab_tools/classification/dist_train_cls.sh \
  research/configs/classification/resnet/resnet18_cifar100/baseline_resplit_cifar100_dataset.py 1 \
  --work-dir ${BASELINE_DIR}
