# Running experiments on induced ReLU models

1. First of all, consider the base configuration file for all scripts. It defines paths, model names, etc.
This file may need to be edited based on the desired experiment.

The following section is a description of all environment variables that are present within this config file:

### NETWORK_TYPE

This the network type that will be used in the association and finetuning routines. Currently, only ResNet18 is supported.

### INDUCER_TO_INDUCED_MAPPING

This a json file describing the mapping between an inducer and the appropriate induced ReLU layers. It should be noted that an inducer can only be an earlier layer with respect to the induced layers (as otherwise it wouldn't make much sense). Any layer that is not present in this json will be left as is (i.e all its ReLU layers will be preserved).

### PROTOTYPES_COUNTS

A SPACE DELIMITED string with the number of prototypes per inducer. The number of prototype counts in this string must be of the same length as the number of inducers in the previous json mapping file. The order of prototypes count in this string must match the order of the inducers in the mapping file.

### NEW_INDUCED_MODEL_NAME

The name of the induced model (based on the inducer -> induced mapping defined earlier) that will be registerd in mmcls registry.

### MMCLS_BACKBONE_DIR

The root directory of mmcls backbones.

### NEW_INDUCED_MODEL_CONFIG_FILE_NAME

The file name (NOT path) of the induced model definition that will be added to mmcls.

### STRUCTURED_MODEL_CONFIG_FILE

The path of the config file of the original structured model.

### FLATTENED_MODEL_CONFIG_FILE

The path to the config file of the flattened model.

### BASELINE_STRUCTURED_CHECKPOINT_PATH

The path to the checkpoint file of the trained original structured model.

### BASELINE_FLATTENED_CHECKPOINT_PATH

The path to the checkpoint file of the trained flattened model. This will be created from the structured checkpoint, so the directory to which this file is saved MUST exist.

### CACHE_DIR

The root directory of the cache directory - the activations cache file will be written here, as well as all computed prototypes and choosing matrices. This directory MUST exist.

### BASE_INDUCED_MODEL_CONFIG_FILE

The path to the config file of the flattened & induced model. This is an output path, so the directory to which this file is saved MUST exist.

### FINETUNE_INDUCED_MODEL_CONFIG_FILE

The path to the config file that is used to define the finetuning routine for the flattened & induced model. This is an output path, so the directory to which this file is saved MUST exist.

### FINETUNE_WORK_DIR

The path to the directory to which all finetuning artifacts will be written. This directory MUST exist.

After populating the config file with our desired values for the environmant variables, we run it:

```shell
source ./induced_config.sh
```

2. Flatten the model of choice. At the moment, only ResNet18 is supported.
Run the following script:

```shell
./flatten_resnet.sh
```

3. Cache ReLU responses based on the flattened model. This will create a file containing the responses across all ReLU layers in the model.

```shell
./cache_relu_activations.sh
```

4. In this stage, we create the checkpoint file for the flattened model, such that when tested on our created flattened model, we would get the same accuracy as the structured one. 
	
	1.	First, we run the conversion routine that creates the checkpoint for the flattened model:
		Run the following script:
		```shell
		./convert_structured_checkpoint.sh
		```

	2.	Validate that the accuracy of the flattened model (and the newly created checkpoint) corresponds to the accuracy of the original structured model by running:
		```shell
		./inference_original_model.sh
		``` 
		Then, run the test routine for the flattened checkpoint:
		```shell
		./inference_flattened_model.sh
		```

		Make sure that the accuracy scores for both runs are identical.

5. Create the model config file for the induced model and add it to mmcls.

```shell
./create_induced_model.sh
```

6. Cluster ReLUs based on the given configuration and create the appropriate permutation matrices.

```shell
./create_permutation_matrices.sh
```

7. Create the appropriate configs for creating the checkpont from which we resume to perform finetuning & the actual finetune routine.

```shell
./create_induced_configs.sh
```

8. Execute the finetune routine for the induced model:

```shell
./finetune_v2.sh
```

At the end of the finetune routine, the accuracy score for the finetuned (induced) model should be shown.
