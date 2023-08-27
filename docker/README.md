# Docker instructions
1. Build the docker:
```shell
docker build -t mmclassification docker/
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/home/code -v $(readlink -f trained_networks):/home/code/trained_networks -it mmclassification:latest  /bin/bash
```
On the 1080:
```shell
docker run --gpus all -v $(pwd):/home/code -v $(readlink -f trained_networks):/home/code/trained_networks -it mmclassification:latest  /bin/bash
```

3. Then reinstall mmpretrain locally (I was too lazy to do it with the docker):
```shell
pip install -e .
pip install scikit-learn
```
4. Hit folder back to go to the main.
```shell
cd ..
```
5. Follow the readme.txt, for example:
```shell

export NUM_PROTOTYPES=1000

./cluster.sh
./permutation_matrices.sh
./greedy.sh

python get_greedily_removed_order.py results/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_prototypes cached_relus/cifar100_with_val/resnet18/${NUM_PROTOTYPES}_most_important_relus/layer_name_to_choosing_matrix/layer_name_to_matrix.pkl research/configs/classification/resnet/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes osi_checkpoints/cifar100/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes trained_networks/classification/resnet18_cifar100_resplit_with_val_${NUM_PROTOTYPES}_prototypes

./finetune.sh
./do_test.sh

python plot_finetuning_result_cifar100_resnet18.py ${NUM_PROTOTYPES}

```
6. Go to: `results/cifar100_with_val/resnet18/10_most_important_prototypes` to enjoy the results:

