mkdir -p results/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_prototypes/
python create_choosing_matrices_fixed.py ${NUM_PROTOTYPES} cached_relus/cifar100_with_val/resnet18/
