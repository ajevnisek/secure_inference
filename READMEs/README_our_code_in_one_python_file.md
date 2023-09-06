# Inducing Layers

## ResNet18 
1) The following bash scripts can run once:
```shell
./flatten_resnet.sh

./cache_relu_activations.sh

./convert_structured_checkpoint.sh

./inference_original_model.sh
./inference_flattened_model.sh
```

They create a flattened version of ResNet18 and help cache relu activations.

2) We write a few scripts to automate our algorithm:
 - `cacher.py` to cache D-ReLU activations.
 - We augmented `flattened_networks_cluster_script`'s `Associator` to support layers which are set to identity.
 - We created `finetuner.py` which:
   - both automates the creation of config files using `finetune_config_file`,
   - and runs the finetune scripts using: `run_finetuning`.

4) An example for a usage combining these scripts are in `cache_cluster_associate_and_finetune.py`:
```shell
python cache_cluster_associate_and_finetune.py
```

## ResNet50 - TBD