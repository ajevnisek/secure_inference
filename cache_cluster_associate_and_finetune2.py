import os, yaml, pickle
import torch
from cacher import cache_activations
from flattened_networks_cluster_script import Associator
from finetuner import create_finetune_config_file, run_finetuning


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



if __name__ == '__main__':
    with open('yaml_configs/split_renset_to_two.yaml', "r") as f:
        network_settings = yaml.load(f, yaml.FullLoader)
    layer_induce_mapping = network_settings['inducer_to_induced']
    inducers_order = network_settings['order']
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    #                                  # CACHE RESNET-18 D-RELU ACTIVATIONS #                                         #
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    # cache
    # cache_activations(model_name='FlattenedResNet18',
    #                   configs_dir='research/configs/classification/flattened_resnet_split_to_two',
    #                   checkpoint_path='/mnt/data/secure_inference_cache/trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/baseline/latest.pth',
    #                   activations_cache_dir='temp_dir/replace_ten_and_then_the_rest/activations',
    #                   activations_output_path='temp_dir/replace_ten_and_then_the_rest/activations/activations_0.pkl',
    #                   is_induce_network=False)

    ###################################################################################################################
    #                            # CLUSTER PROTOTYPES AND ASSOCIATE DEEP RELUS TO PROTOTYPES #                        #
    ###################################################################################################################
    # cluster & associate
    inducer_to_induced = {inducers_order[0]: layer_induce_mapping[inducers_order[0]]}
    # associator = Associator('temp_dir/replace_ten_and_then_the_rest/activations/activations_0.pkl',
    #                         inducer_to_induced, [network_settings['num_prototypes'][0]],
    #                         'temp_dir/replace_ten_and_then_the_rest/associations',
    #                         recluster=True)
    # associator.run()
    ###################################################################################################################
    #                                        # RETRAIN NEURAL NETWORK #                                               #
    ###################################################################################################################
    # duplicate checkpoint and fix running parameters to enable fine-tuning:
    previous_checkpoint = '/mnt/data/secure_inference_cache/trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/baseline/latest.pth'
    new_checkpoint = 'temp_dir/replace_ten_and_then_the_rest/checkpoints/checkpoint_0.pth'
    fix_checkpoint(previous_checkpoint, new_checkpoint, epochs_to_finetune=5)

    # create config for fine-tuning:
    create_finetune_config_file('ResNetInduceInParts_0',
                                'research/configs/classification/flattened_resnet_split_to_two/finetune_0.py',
                                is_induce_network=True,
                                choosing_matrices_full_path='temp_dir/replace_ten_and_then_the_rest/associations/layer_name_to_choosing_matrix/layer_name_to_matrix.pkl',
                                previous_checkpoint=new_checkpoint)

    # # finetune
    # run_finetuning('research/configs/classification/flattened_resnet_split_to_two/finetune_0.py',
    #                'temp_dir/replace_ten_and_then_the_rest/trained_networks/finetune_0')

    ###################################################################################################################
    #                                  # CACHE MODIFIED RESNET-18 D-RELU ACTIVATIONS #                                #
    ###################################################################################################################
    # cache
    cache_activations('ResNetInduceInParts_0',
                      'research/configs/classification/flattened_resnet_split_to_two',
                      'temp_dir/replace_ten_and_then_the_rest/trained_networks/finetune_0/latest.pth',
                      'temp_dir/replace_ten_and_then_the_rest/activations',
                      'temp_dir/replace_ten_and_then_the_rest/activations/activations_1.pkl',
                      is_induce_network=True,
                      choosing_matrices_full_path='temp_dir/replace_ten_and_then_the_rest/associations/layer_name_to_choosing_matrix/layer_name_to_matrix.pkl')
    ###################################################################################################################
    #                            # CLUSTER PROTOTYPES AND ASSOCIATE DEEP RELUS TO PROTOTYPES #                        #
    ###################################################################################################################
    # cluster & associate
    inducer_to_induced = {inducers_order[1]: layer_induce_mapping[inducers_order[1]]}
    associator = Associator('temp_dir/replace_ten_and_then_the_rest/activations/activations_1.pkl',
                            inducer_to_induced, [network_settings['num_prototypes'][1]],
                            'temp_dir/replace_ten_and_then_the_rest/associations',
                            recluster=True,
                            layer_name_to_matrix_pickle_filename='layer_name_to_matrix_after_first_finetune.pkl')
    associator.run()
    ###################################################################################################################
    #                                        # RETRAIN NEURAL NETWORK #                                               #
    ###################################################################################################################
    # fix checkpoint
    previous_checkpoint = 'temp_dir/replace_ten_and_then_the_rest/trained_networks/finetune_0/latest.pth'
    new_checkpoint = 'temp_dir/replace_ten_and_then_the_rest/checkpoints/checkpoint_1.pth'
    fix_checkpoint(previous_checkpoint, new_checkpoint, epochs_to_finetune=200)

    # fix layer_name_to_matrix_pickle:
    concatenate_pickles_to_new_pickle_path('temp_dir/replace_ten_and_then_the_rest/associations/layer_name_to_choosing_matrix/layer_name_to_matrix.pkl',
                                           'temp_dir/replace_ten_and_then_the_rest/associations/layer_name_to_choosing_matrix/layer_name_to_matrix_after_first_finetune.pkl',
                                           'temp_dir/replace_ten_and_then_the_rest/associations/layer_name_to_choosing_matrix/layer_name_to_matrix_merged.pkl')

    # create config for fine-tuning:
    create_finetune_config_file('ResNetInduceInParts_1',
                                'research/configs/classification/flattened_resnet_split_to_two/finetune_1.py',
                                is_induce_network=True,
                                choosing_matrices_full_path='temp_dir/replace_ten_and_then_the_rest/associations/layer_name_to_choosing_matrix/layer_name_to_matrix_merged.pkl',
                                previous_checkpoint=new_checkpoint)

    # finetune
    run_finetuning('research/configs/classification/flattened_resnet_split_to_two/finetune_1.py',
                   'temp_dir/replace_ten_and_then_the_rest/trained_networks/finetune_1')
