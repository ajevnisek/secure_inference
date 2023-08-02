import argparse
import time
import pickle
import os.path as osp

import torch
import numpy as np
import sklearn.metrics as metrics

from tqdm import tqdm


def farthest_point_sampling(points, num_samples, metric: float = 2):
    num_points = points.shape[0]
    distances = torch.full((num_points, 1), torch.inf)
    samples = torch.zeros(num_samples)
    farthest_idx = torch.randint(0, num_points, (1, )).item()

    for i in tqdm(range(num_samples)):
        samples[i] = farthest_idx
        current_point = points[farthest_idx]
        distances_to_current_point = torch.cdist(points, current_point.unsqueeze(0), p=metric)
        distances = torch.tensor([torch.min(x, y) for x, y in zip(distances, distances_to_current_point)])
        farthest_idx = torch.argmax(distances).item()

    return samples


def parse_args():
    parser = argparse.ArgumentParser(description='find correlated ReLUs')
    parser.add_argument('activations', help='input activations',
                        default='cached_relus/cifar100/resnet18/activations.pkl')
    parser.add_argument('outdir', help='cache output directory',
                        default='cached_relus/cifar100/resnet18')
    return parser.parse_args()


def cluster_relus_in_first_layer(args):

    with open(args.activations, 'rb') as f:
        activations = pickle.load(f)

    for k in activations:
        print(k, activations[k].shape)

    relus_first_layer = activations['act1']
    print(f"The DReLUs stats from the first layer, has shape: {relus_first_layer.shape}")
    num_protoypes = 1000
    start_time = time.time()
    prototypes_indices = farthest_point_sampling(
        relus_first_layer.flatten(1).T.float(), num_protoypes, metric=0)
    with open(osp.join(args.outdir, '1K_prototypes_indices_p=0.pkl'),
              'wb') as f:
        pickle.dump(prototypes_indices, f)
    prototypes = torch.index_select(relus_first_layer.flatten(1).T, 0,
                                    prototypes_indices.int()).T

    print(f"")
    with open(osp.join(args.outdir,
                       '1K_prototypes_p=0.pkl'), 'wb') as f:
        pickle.dump(prototypes, f)
    end_time = time.time()
    print(f"FSP took: {end_time - start_time} seconds")
    print(f"We use Farthest point sampling to cluster the ReLUs to {num_protoypes}.")
    print(f"The prototypes shape is: {prototypes.shape}")
    return prototypes, activations

def associate_relus_to_prototypes(activations, prototypes, outdir):

    layer_1_0_relu_1 = activations['layer1[0].relu_1']
    D, H, W = layer_1_0_relu_1.shape[1:]
    layer_1_0_relu_1 = layer_1_0_relu_1.flatten(1).T
    print(f'Now for each ReLU, let\'s calculate the prototype it is most '
          f'correlated with..')
    print(f"Each prototype and ReLU is with dimension: {prototypes.shape[-1]}")
    print(f"Associating {layer_1_0_relu_1.shape[0]} ReLUs with"
          f" {prototypes.shape[0]} prototypes...")


    for layer_name in [
        'layer1[0].relu_1', 'layer1[0].relu_2',
        'layer1[1].relu_1', 'layer1[1].relu_2',
        'layer2[0].relu_1', 'layer2[0].relu_2',
        'layer2[1].relu_1', 'layer2[1].relu_2',
        'layer3[0].relu_1', 'layer3[0].relu_2',
        'layer3[1].relu_1', 'layer3[1].relu_2',
        'layer4[0].relu_1', 'layer4[0].relu_2',
        'layer4[1].relu_1', 'layer4[1].relu_2',
    ]:
        start_time = time.time()
        layer_responses = activations[layer_name].flatten(1).T
        _chunk_size = 1000
        closest_prototype_chunks = []
        for i in tqdm(range(layer_responses.shape[0] // _chunk_size + 1)):
            closest_prototype_chunks.append(torch.from_numpy(
                metrics.pairwise_distances_argmin(
                    layer_responses[i * _chunk_size: (i + 1) * _chunk_size],
                    prototypes.T, metric='hamming')))
        end_time = time.time()
        print(
            f"associating ReLUs with prototypes for layer {layer_name} took"
            f" {end_time - start_time} seconds")
        closest_prototypes = closest_prototype_chunks
        with open(osp.join(outdir,
                           f'closest_prototypes_{layer_name}.pkl'), 'wb') as f:
            pickle.dump(closest_prototypes, f)


def main():
    args = parse_args()
    prototypes, activations = cluster_relus_in_first_layer(args)
    associate_relus_to_prototypes(activations, prototypes, args.outdir)


if __name__ == '__main__':
    main()
