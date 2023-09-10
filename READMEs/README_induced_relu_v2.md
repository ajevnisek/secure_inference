# Running experiments on induced ReLU models

1. First of all, consider the base configuration file for all scripts. It defines paths, model names, etc.
This file may need to be edited based on the desired experiment.

The following section is a description of all environment variables that are present within this config file:

### YAML_CONFIG

A path to the yaml config file that contains the pipeline data.

### EXPERIMENT_NAME

A user chosen name for the current experiment. Directories will be created based on this name to differentiate experiments.

### EXPERIMENTS_DIR

The root directory for all experiments. All intermediate results will be saved in this directory.
This directory MUST exist.

### CONFIG_DIR

The root directory for all config files. The base directory for all configs must reside here, so that any config files will be able to import common configurations.

### NETWORK_TYPE

This the network type that will be used in the association and finetuning routines. Currently, only ResNet18 is supported.

### NEW_INDUCED_MODEL_NAME

The name of the induced model (based on the yaml config file) that will be registerd in mmcls registry.

### NEW_INDUCED_MODEL_CONFIG_FILE_NAME

The file name (NOT path) of the induced model definition that will be added to mmcls.

### MMCLS_BACKBONE_DIR

The root directory of mmcls backbones.

### STRUCTURED_MODEL_CONFIG_FILE

The path of the config file of the original structured model.

### FLATTENED_MODEL_CONFIG_FILE

The path to the config file of the flattened model.

### BASELINE_STRUCTURED_CHECKPOINT_PATH

The path to the checkpoint file of the trained original structured model.

### BASELINE_FLATTENED_CHECKPOINT_PATH

The path to the checkpoint file of the trained flattened model. This will be created from the structured checkpoint, so the directory to which this file is saved MUST exist.

After populating the config file with our desired values for the environmant variables, we run it:

```shell
source ./induced_config.sh
```

2. Flatten the model of choice. At the moment, only ResNet18 is supported.
Run the following script:

```shell
./flatten_resnet.sh
```

3. In this stage, we create the checkpoint file for the flattened model, such that when tested on our created flattened model, we would get the same accuracy as the structured one. 
	
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

4. Create the various induced model files based on the YAML config file:

```shell
./iterate_inducers_build_networks.sh
```

5. Finally, we run the main routine which clusters, associates and finetunes the model based on the YAML configuration file.

```shell
./process_inducers.sh
```

At the end of the finetune routine, the accuracy score for the finetuned (induced) model should be shown.
