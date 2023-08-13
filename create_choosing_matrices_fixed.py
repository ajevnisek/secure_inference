import os
import pickle
import argparse
import os.path as osp
from functools import reduce
from typing import List, Tuple

import torch.nn


def calculate_choosing_matrix(closest_prototypes: torch.Tensor,
                              prototypes_indices_in_first_relu_layer: torch.Tensor,
                              flattened_first_relu_layer_shape: int):
    """
    Calculates the choosing matrix based on closest prototypes evaluated
    earlier, the prototypes indices.

    Args:
        closest_prototypes (torch.Tensor): A tensor representing the closest
                                           prototypes.
        prototypes_indices_in_first_relu_layer (torch.Tensor):
        A tensor representing the indices of prototypes in the first ReLU layer.
        flattened_first_relu_layer_shape (int): The shape of the first ReLU
        layer.

    Returns:
        torch.Tensor: The choosing matrix.


    The permutation matrix which we call here choosing matrix (and we might
    as well have called extracting matrix or permutation matrix) is a
    sparse matrix where each row is a one-hot vector with the shape of the
    first ReLUs layer, flattened. The matrix has rows a the number of ReLUs
    in the current layer, flattened. Each row is a one-hot vector such that 1
    indicates the prototype index associated with this ReLU.

    Note that we need to convert the prototype indices from the cluster index
    back to the ReLU layer indices.

    The choosing matrix is transposed to enable efficient multiplication from
    the right.
    """
    flattened_relus_shape = closest_prototypes.shape[0]
    where_to_look = closest_prototypes
    indices = [torch.index_select(prototypes_indices_in_first_relu_layer, 0,
                                  where_to_look).int().tolist(),
               list(range(flattened_relus_shape))]
    # choosing matrix.shape = 8192x65536
    # choosing matrix transposed such that we can multiply it from right:
    # (Bx65536) @ (65536x8192)
    choosing_matrix = torch.sparse_coo_tensor(
        indices, [1.] * flattened_relus_shape,
        (flattened_first_relu_layer_shape, flattened_relus_shape))

    # choosing_matrix = choosing_matrix.to_dense()
    return choosing_matrix


def create_all_choosing_matrices(args):
    with open(osp.join(args.cache_dir, 'activations.pkl'), 'rb') as f:
        activations = pickle.load(f)

    relus_first_layer = activations['act1']
    print(f"Loaded ReLU activations for the first layer...")
    print(f"ReLUs in the first layer have shape: {relus_first_layer.shape}...")

    prototypes_dir = osp.join(args.cache_dir, f'{args.num_prototypes}_most_important_relus')
    with open(osp.join(prototypes_dir, f'{args.num_prototypes}_prototypes_indices_p=0.pkl'), 'rb') as f:
        prototypes_indices = pickle.load(f)
    print(f"We clustered protocols for the first layer with FPS and got "
          f"{prototypes_indices.shape[0]} centers...")

    layer_name_to_matrix = {}
    for layer_index in [1, 2, 3, 4]:
        for sub_index in [0, 1]:
            for act_index in [1, 2]:
                layer_name = f'layer{layer_index}[{sub_index}].relu_{act_index}'
                name = f'closest_prototypes_{layer_name}.pkl'
                with open(os.path.join(prototypes_dir, name), 'rb') as f:
                    closest_prototypes_for_layer_name = torch.cat(
                        pickle.load(f))
                print(
                    f'We calculated prototypes matching to ReLU: '
                    f'{layer_name}, '
                    f'the closest-prototypes shape is: '
                    f'{closest_prototypes_for_layer_name.shape}')
                choosing_matrix = calculate_choosing_matrix(
                    closest_prototypes_for_layer_name, prototypes_indices,
                    relus_first_layer.flatten(1).shape[-1])
                layer_name_to_matrix[layer_name] = choosing_matrix
                os.makedirs(os.path.join(prototypes_dir,
                                         'layer_name_to_choosing_matrix'),
                            exist_ok=True)
                with open(osp.join(prototypes_dir, 'layer_name_to_choosing_matrix',
                                   f'{layer_name}_to_matrix.pkl'), 'wb') as f:
                    pickle.dump(choosing_matrix, f)
    with open(osp.join(prototypes_dir, 'layer_name_to_choosing_matrix',
                       'layer_name_to_matrix.pkl'), 'wb') as f:
        pickle.dump(layer_name_to_matrix, f)


def parse_args():
    parser = argparse.ArgumentParser(description='create choosing ('
                                                 'permutation) matrices')
    parser.add_argument('num_prototypes', type=int, help='num prototypes that used in the ReLU association stage')
    parser.add_argument('cache_dir', type=str, help='cache directory root',
                        default='cached_relus/cifar100_with_val/resnet18/')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_all_choosing_matrices(args)
