mkdir -p cached_relus/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_relus
python find_correlated_relus_fixed.py ${NUM_PROTOTYPES} cached_relus/cifar100_with_val/resnet18/activations.pkl cached_relus/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_relus
