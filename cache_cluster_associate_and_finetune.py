import os, yaml, pickle
import shutil
import torch
import argparse
from cacher import cache_activations
from flattened_networks_cluster_script import Associator
from finetuner import create_finetune_config_file, run_finetuning

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_model_name', type=str, help='the flattened model we work with')
    parser.add_argument('induced_model_name', type=str, help='the induced model name that we previously registerd with mmcls')
    parser.add_argument('base_checkpoint_path', type=str, help='the path to the flattened base model\'s checkpoint')
    parser.add_argument('experiment_name', type=str, help='the name of the current experiment')
    parser.add_argument('experiments_dir', type=str, help='experiments base dir')
    parser.add_argument('configs_dir', type=str, help='root dictory of configs')
    parser.add_argument("pipeline_yaml", type=str, help='a path to the yaml file describing the pipeline to perform')

    return parser.parse_args()

"""
WARNING: Do NOT call this function with a different parameter than the default one for 'epochs_to_finetune'. 
The current implementation does not account for learning rates when setting the epoch number for the saved checkpoint, so in this case,
what happens is that when we set a high epoch count (say 150 and up), the initial lr will be too high for a finetune routine.
So at this point, the implementation assumes that we start at the 180th epoch (and thus, initial lr would be 8e-4). The lr scheduler is
defined in the config pattern for the finetuning routine in finetuner.py.
"""
def fix_checkpoint(previous_checkpoint: str, new_checkpoint: str, epochs_to_finetune: int = 20):
    d = torch.load(previous_checkpoint)
    original_num_epochs = d['meta']['epoch']
    d['meta']['epoch'] = int(original_num_epochs - epochs_to_finetune)
    d['meta']['iter'] = int(d['meta']['iter'] * d['meta']['epoch'] / original_num_epochs)
    torch.save(d, new_checkpoint)

def concatenate_pickles_to_new_pickle_path(prev_pickle_path, curr_pickle_path, new_pickle_path):
    def path_to_dict(path):

        with open(path, 'rb') as f:
            d = pickle.load(f)
        return d
    prev = path_to_dict(prev_pickle_path)
    curr = path_to_dict(curr_pickle_path)
    curr.update(prev)
    with open(new_pickle_path, 'wb') as f:
        pickle.dump(curr, f)
        
def do_process_inducer(path_to_activations_cache, inducer_to_induced, num_prototypes, outputs_dir, recluster):
    associator = Associator(path_to_activations_cache=path_to_activations_cache,
                            inducer_to_induced=inducer_to_induced,
                            num_prototypes=num_prototypes,
                            outputs_dir=outputs_dir,
                            recluster=recluster)

    associator.run()
    associator.cache = None

def process_inducers(settings, base_model_name, induced_model_name, experiment_name, experiments_dir, configs_root_dir, model_checkpoint_path):

    layer_inducer_mapping = settings['inducer_to_induced']
    inducers_order = settings['order']
    finetune_epochs = settings['epochs']

    activations_dir = os.path.join(experiments_dir, experiment_name, "activations")
    associations_dir = os.path.join(experiments_dir, experiment_name, "associations")
    combined_associations_dir = os.path.join(experiments_dir, experiment_name, "associations", "combined")
    permutation_matrices_dir = os.path.join(combined_associations_dir, 'layer_name_to_choosing_matrix')
    checkpoints_dir = os.path.join(experiments_dir, experiment_name, "checkpoints")
    results_dir = os.path.join(experiments_dir, experiment_name, "results")

    os.makedirs(activations_dir, exist_ok=True)
    os.makedirs(associations_dir, exist_ok=True)
    os.makedirs(combined_associations_dir, exist_ok=True)
    os.makedirs(permutation_matrices_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    experiment_configs_dir = os.path.join(configs_root_dir, experiment_name)
    os.makedirs(experiment_configs_dir, exist_ok=True)

    # At the first iteration, we work with the base flattened model (as nothing has been induced yet)

    current_model_name = base_model_name
    current_checkpoint_path = model_checkpoint_path
    is_induced_network = False
    current_choosing_matrices_full_path = ''
    
    """
    current_model_name = f'{induced_model_name}_2'
    current_checkpoint_path = os.path.join(results_dir, 'finetune_2', 'latest.pth')
    is_induced_network = True
    current_choosing_matrices_full_path = os.path.join(combined_associations_dir, 'layer_name_to_choosing_matrix', 'layer_name_to_matrix.pkl')
    """
    
    for i in range(0, len(layer_inducer_mapping)):

        # Each activation map in each step will be saved in a separate file tagged by the step index
        current_activations_cache_path = os.path.join(activations_dir, f"activations_{i}.pkl")

        cache_activations(model_name=current_model_name,
                            configs_dir=experiment_configs_dir,
                            checkpoint_path=current_checkpoint_path,
                            activations_cache_dir=activations_dir,
                            activations_output_path=current_activations_cache_path,
                            is_induced_network=is_induced_network,
                            choosing_matrices_full_path=current_choosing_matrices_full_path)
                            
        """
        This will create the permutation matrices, each such matrix will be placed in its own directory, tagged by the step index.
        Each time we only induce a single layer (but possibly already induced some layers before that, so we always work with the
        updated model).
        """
        inducer_to_induced = {inducers_order[i] : layer_inducer_mapping[inducers_order[i]]}

        current_associations_dir = os.path.join(associations_dir, f'associations_{i}')
        os.makedirs(current_associations_dir, exist_ok=True)
        
        

        do_process_inducer(path_to_activations_cache=current_activations_cache_path,
                            inducer_to_induced=inducer_to_induced,
                            num_prototypes=[settings['num_prototypes'][0]],
                            outputs_dir=current_associations_dir,
                            recluster=True)


        """
        Here we create a new checkpoint depending on the checkpoint from the last phase. It is created with modified metadata so that
        it can be finetuned starting from a later stage (which would adjust learning rate and scheduling).
        """
        new_checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_{i}.pth')

        fix_checkpoint(current_checkpoint_path, new_checkpoint_path)

        """
        In the multi stage induce procedure, the matrices must be combined to contain all permutation matrices for the layers computed so far.
        Then, we can create the appropriate config file for the model by plugging in the appropriate permutation matrices. Using this config file,
        we can finetune the network and arrive at a more optimized model.
        """
        current_model_name = f'{induced_model_name}_{i}'

        current_choosing_matrices_full_path = os.path.join(current_associations_dir, 'layer_name_to_choosing_matrix', 'layer_name_to_matrix.pkl')
        combined_choosing_matrices_path = os.path.join(combined_associations_dir, 'layer_name_to_choosing_matrix', 'layer_name_to_matrix.pkl')

        if not is_induced_network:
            shutil.copy(current_choosing_matrices_full_path, combined_choosing_matrices_path)
        else:
            concatenate_pickles_to_new_pickle_path(combined_choosing_matrices_path, current_choosing_matrices_full_path, combined_choosing_matrices_path)
        
        # The effective matrices path is the combined one (even if it's just the matrices for the first iteration)
        current_choosing_matrices_full_path = combined_choosing_matrices_path

        finetune_config_path = os.path.join(experiment_configs_dir, f'finetune_{i}.py')

        # From now on, we work with induced networks as the permutation matrices have been defined for the induced model
        is_induced_network = True
        
        # At the last iteration, we need to indicate that the finetuning must be done on the entire train set
        is_last_iteration = (i == (len(layer_inducer_mapping) - 1))

        create_finetune_config_file(model_name=current_model_name,
                                    output_path=finetune_config_path,
                                    num_epochs=int(finetune_epochs[i]),
                                    is_induced_network=is_induced_network,
                                    choosing_matrices_full_path=combined_choosing_matrices_path,
                                    previous_checkpoint=new_checkpoint_path,
                                    use_entire_train_set=is_last_iteration)

        """
        Each finetune's result files will reside in a different directory. Now we can start finetuning and saving the finetune result for
        the initial checkpoint for the next stage (if applicable).
        """
        finetune_results_dir = os.path.join(results_dir, f'finetune_{i}')
        os.makedirs(finetune_results_dir, exist_ok=True)

        run_finetuning(finetune_config_file=finetune_config_path, results_dir=finetune_results_dir)

        current_checkpoint_path = os.path.join(finetune_results_dir, 'latest.pth')


def main(args):

    with open(args.pipeline_yaml, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    process_inducers(settings=cfg, 
                    base_model_name=args.base_model_name,
                    induced_model_name=args.induced_model_name, 
                    experiment_name=args.experiment_name, 
                    experiments_dir=args.experiments_dir,
                    configs_root_dir=args.configs_dir, 
                    model_checkpoint_path=args.base_checkpoint_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
