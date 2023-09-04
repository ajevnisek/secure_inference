# How to use "run all" scripts?
1. Decide which inducer <-> induced relationship you'd like to implement and create a json to reflect it and place it in `json_configs`:

For example:
```json
{
    "relu0": [
        "relu0",
        "ResLayer0_BasicBlockV20_relu_1"
    ]
}
```

2. Create a configs file and place it under `configs_our_framework`. Make sure to change:
 - INDUCER_TO_INDUCED_MAPPING
 - NEW_INDUCED_MODEL_NAME
 - NEW_INDUCED_MODEL_CONFIG_FILE_NAME
 - CACHE_DIR
 - BASE_INDUCED_MODEL_CONFIG_FILE
 - FINETUNE_INDUCED_MODEL_CONFIG_FILE
 - FINETUNE_WORK_DIR

`source` the config file and make sure the following exist:
```shell
source configs_our_framework/<your_config_file.sh>
mkdir -p $(dirname ${BASELINE_FLATTENED_CHECKPOINT_PATH})
mkdir -p $(dirname ${CACHE_DIR})
mkdir -p $(dirname ${BASE_INDUCED_MODEL_CONFIG_FILE})
mkdir -p $(dirname ${FINETUNE_INDUCED_MODEL_CONFIG_FILE})
mkdir -p ${FINETUNE_WORK_DIR}
```

For example, see: `configs_our_framework/induced_config_one_layer.sh`

3. Create a run all script in `run_all_scripts` and change config file it calls.
For example, see: `run_all_scripts/run_all_self_induce_one_layer.sh`

Although this has code duplication, it has reproducability benefits.