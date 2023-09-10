# A path to the yaml config file that contains the pipeline data.
export YAML_CONFIG="experiments/experiment4/experiment4.yaml"

# A user chosen name for the current experiment. Directories will be created based on this name to
# differentiate experiments.
export EXPERIMENT_NAME="Experiment4"

# The root directory for all experiments. All intermediate results will be saved in this directory.
# This directory MUST exist.
export EXPERIMENTS_DIR="/mnt/data/secure_inference_cache/experiments/"

# The root directory for all config files. The base directory for all configs must reside here, so that any
# config files will be able to import common configurations.
export CONFIG_DIR="research/configs/classification"

# This the network type that will be used in the association and finetuning routines. Currently, only ResNet18 is supported.
export NETWORK_TYPE="ResNet18"

# The name of the induced model (based on the yaml config file) that will be registerd in mmcls registry.
export NEW_INDUCED_MODEL_NAME=ResNetInducedExperiment4

# The file name (NOT path) of the induced model definition that will be added to mmcls.
export NEW_INDUCED_MODEL_CONFIG_FILE_NAME=resnet18_induced_experiment4.py

# The root directory of mmcls backbones.
export MMCLS_BACKBONE_DIR=mmpretrain/mmcls/models/backbones/

# The path of the config file of the original structured model.
export STRUCTURED_MODEL_CONFIG_FILE=research/configs/classification/resnet/resnet18_cifar100_resplit_with_val/baseline_resplit_cifar100_dataset.py

# The path to the config file of the flattened model.
export FLATTENED_MODEL_CONFIG_FILE=research/configs/classification/flattened_resnet/baseline_resplit_cifar100_dataset_flattened_resnet18.py

# The path to the checkpoint file of the trained original structured model.
export BASELINE_STRUCTURED_CHECKPOINT_PATH=/mnt/data/secure_inference_cache/trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth

# The path to the checkpoint file of the trained flattened model. This will be created from the structured checkpoint, so the directory to which this file is saved MUST exist.
export BASELINE_FLATTENED_CHECKPOINT_PATH=/mnt/data/secure_inference_cache/trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/baseline/latest.pth
