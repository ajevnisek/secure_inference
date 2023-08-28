import argparse
import json
import time
import pickle
import os.path as osp

import torch
import numpy as np
import sklearn.metrics as metrics

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='find correlated ReLUs')
    parser.add_argument('inducer_to_induced_json', type=str, help='path to a json holding the mapping between the '
                                                                  'inducer and induced relu layers')
    parser.add_argument('num_prototypes', nargs='+', type=int, help='a list of the number of prototypes for clustering')
    parser.add_argument('activations_cahce', help='activations cache',
                        default='cached_relus/cifar100_with_val/flattened_resnet18/activations.pkl')
    parser.add_argument('outdir', help='cache output directory',
                        default='cached_relus/cifar100_with_val/flattened_resnet18')
    return parser.parse_args()





class cluster:
    def __init__(self, path_to_activations_cache: str, inducer_to_induced: dict, num_prototypes: list, outputs_dir: str):
        assert len(num_prototypes) == len(inducer_to_induced), "Explicitly specify the number of prototypes for each inducer layer."
        self.inducer_to_induced = inducer_to_induced
        self.num_prototypes = num_prototypes

        with open(path_to_activations_cache, 'rb') as f:
            self.cache = pickle.load(f)
        self.outputs_dir = outputs_dir

    def cluster_one_inducer(self, inducer_name: str, num_prototypes: int):
        relus_first_layer = self.cache[inducer_name]
        print(f"The DReLUs stats from layer: {relus_first_layer}, has shape: {relus_first_layer.shape}")
        start_time = time.time()
        prototypes_indices = self.farthest_point_sampling(
            relus_first_layer.flatten(1).T.float(), num_prototypes, metric=0)
        with open(osp.join(self.outputs_dir, f'{inducer_name}_{num_prototypes}_prototypes_indices.pkl'),
                  'wb') as f:
            pickle.dump(prototypes_indices, f)
        prototypes = torch.index_select(relus_first_layer.flatten(1).T, 0,
                                        prototypes_indices.int()).T

        print(f"")
        with open(osp.join(self.outputs_dir,
                           f'{inducer_name}_{num_prototypes}_prototypes.pkl'), 'wb') as f:
            pickle.dump(prototypes, f)
        end_time = time.time()
        print(f"FPS took: {end_time - start_time} seconds")
        print(f"We use Farthest point sampling to cluster the ReLUs to {num_prototypes}.")
        print(f"The prototypes shape is: {prototypes.shape}")

    def associated_relus_to_cluster_center(self, inducer_name: str, num_prototypes: int):
        layer_1_0_relu_1 = self.cache[inducer_name]
        D, H, W = layer_1_0_relu_1.shape[1:]
        layer_1_0_relu_1 = layer_1_0_relu_1.flatten(1).T
        prototypes_path = osp.join(self.outputs_dir, f'{inducer_name}_{num_prototypes}_prototypes.pkl')
        with open(prototypes_path, 'rb') as f:
            prototypes = pickle.load(f)
        print(f'Now for each ReLU, let\'s calculate the prototype it is most '
              f'correlated with..')
        print(f"Each prototype and ReLU is with dimension: {prototypes.shape[-1]}")
        print(f"Associating {layer_1_0_relu_1.shape[0]} ReLUs with"
              f" {prototypes.shape[0]} prototypes...")

        for layer_name in self.inducer_to_induced[inducer_name]:
            start_time = time.time()
            layer_responses = self.cache[layer_name].flatten(1).T
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
            with open(osp.join(self.outputs_dir,
                               f'closest_prototypes_{layer_name}.pkl'), 'wb') as f:
                pickle.dump(closest_prototypes, f)

    @staticmethod
    def farthest_point_sampling(points, num_samples, metric: float = 2):
        num_points = points.shape[0]
        distances = torch.full((num_points, 1), torch.inf)
        samples = torch.zeros(num_samples)
        farthest_idx = torch.randint(0, num_points, (1,)).item()

        if num_samples == num_points:
            return torch.range(0, num_samples - 1)

        for i in tqdm(range(num_samples)):
            samples[i] = farthest_idx
            current_point = points[farthest_idx]
            distances_to_current_point = torch.cdist(points, current_point.unsqueeze(0), p=metric)
            distances = torch.tensor([torch.min(x, y) for x, y in zip(distances, distances_to_current_point)])
            farthest_idx = torch.argmax(distances).item()

        return samples

    def run(self):
        for inducer, num_prototypes in zip(self.inducer_to_induced, self.num_prototypes):
            self.cluster_one_inducer(inducer, num_prototypes)
            self.associated_relus_to_cluster_center(inducer, num_prototypes)


def main():
    args = parse_args()
    with open(args.inducer_to_induced_json) as f:
        inducer_to_induced = json.load(f)
    clusterer = cluster(args.activations_cahce, inducer_to_induced, args.num_prototypes, args.outdir)
    clusterer.run()


if __name__ == '__main__':
    main()
