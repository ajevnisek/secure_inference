1. First, define the number of prototypes to use for clustering as an environment variable:

For instance:

export NUM_PROTOTYPES=500

2. Cache ReLUs with farthest point sampling:

./cluster.sh

3. Create the permutation matrices:

./permutation_matrices.sh

4. Run the greedy algorithm:

./greedy.sh

5. Fix the config files:

python get_greedily_removed_order.py results/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_prototypes cached_relus/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_relus/layer_name_to_choosing_matrix/layer_name_to_matrix.pkl research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes osi_checkpoints/cifar100/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes trained_networks/classification/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes

6. Finetune the produced networks:

./finetune.sh

7. Test the networks:

./do_test.sh

8. Plot accuracy graphs:

python plot_finetuning_result_cifar100_resnet18.py ${NUM_PROTOTYPES}



