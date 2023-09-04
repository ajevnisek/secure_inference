# This the network type that will be used in the association and finetuning routines. Currently, only ResNet18 is supported.
export NETWORK_TYPE="ResNet18"

# This a json file describing the mapping between an inducer and the appropriate induced ReLU layers. It should be noted that an inducer can only be an earlier
# layer with respect to the induced layers (as otherwise it wouldn't make much sense). Any layer that is not present in this json will be left as is (i.e all its ReLU layers will be preserved).
export INDUCER_TO_INDUCED_MAPPING=json_configs/flattened_resnet18_one_layer_induces_ten_layers.json

# This controls ig the assignment scripts calculates relus clustering before building association matrices.
export IS_RECLUSTER="TRUE"

# A SPACE DELIMITED string with the number of prototypes per inducer. The number of prototype counts in this string must be of the same length as the number of inducers in the previous json
# mapping file. The order of prototypes count in this string must match the order of the inducers in the mapping file.
export PROTOTYPES_COUNTS="1000"

# The name of the induced model (based on the inducer -> induced mapping defined earlier) that will be registerd in mmcls registry.
export NEW_INDUCED_MODEL_NAME=ResNet18OneLayerWithSelfInduceToTenLayers

# The file name (NOT path) of the induced model definition that will be added to mmcls. 
export NEW_INDUCED_MODEL_CONFIG_FILE_NAME=resnet18_one_layer_induces_ten_layers_with_self_induce.py

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

# The root directory of the cache directory - the activations cache file will be written here, as well as all computed prototypes and choosing matrices.  This directory MUST exist.
export CACHE_DIR=cached_relus/cifar100_with_val/flattened_resnet18/

# The path to the config file of the flattened & induced model. This is an output path, so the directory to which this file is saved MUST exist.
export BASE_INDUCED_MODEL_CONFIG_FILE=research/configs/classification/flattened_resnet/replacing_first_ten_layersresplit_cifar100_dataset.py

# The path to the config file that is used to define the finetuning routine for the flattened & induced model. This is an output path, so the directory to which this file is saved MUST exist.
export FINETUNE_INDUCED_MODEL_CONFIG_FILE=research/configs/classification/flattened_resnet/replacing_first_ten_layerswith_self_induce_resplit_cifar100_dataset.py

# The path to the directory to which all finetuning artifacts will be written. This directory MUST exist.
export FINETUNE_WORK_DIR=trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/replacing_first_ten_layerswith_self_induce/
