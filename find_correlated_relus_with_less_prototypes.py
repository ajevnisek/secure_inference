import os
import argparse
import time
import pickle
import os.path as osp

import torch
import numpy as np
import sklearn.metrics as metrics

from tqdm import tqdm
import time
from tqdm import tqdm
import sklearn.metrics as metrics


def extract_top_k_most_frequently_associated_prototypes(k: int = 1000):

    with open("cached_relus/cifar100_with_val/resnet18/activations.pkl", "rb") as f:
        activations = pickle.load(f)
    import os.path as osp

    with open(
        osp.join("cached_relus/cifar100_with_val/resnet18/", "1K_prototypes_p=0.pkl"),
        "rb",
    ) as f:
        prototypes_values = pickle.load(f)

    with open(
        "cached_relus/cifar100_with_val/resnet18/" "1K_protorypes_usage_hist.pkl", "rb"
    ) as f:
        all_stats = pickle.load(f)

    prots = list(all_stats.keys())
    prots.sort(key=lambda x: all_stats[x])
    usage = list(all_stats.values())
    usage.sort()
    prototypes_indices_path = (
        "cached_relus/cifar100_with_val/resnet18/1K_prototypes_indices_p=0.pkl"
    )
    with open(prototypes_indices_path, "rb") as f:
        prototypes_indices = pickle.load(f)

    get_index = lambda tensor, x: ((tensor == x).nonzero(as_tuple=True)[0])
    prototypes_subset_indices = np.ravel_multi_index(
        (
            [x[0] for x in prots[:k]],
            [x[1] for x in prots[:k]],
            [x[2] for x in prots[:k]],
        ),
        (32, 32, 64),
    )
    locations_of_selected_prototypes = [
        get_index(prototypes_indices, x) for x in prototypes_subset_indices
    ]
    prots_subset = torch.index_select(
        prototypes_values.T, 0, torch.tensor(locations_of_selected_prototypes)
    )

    return activations, prots_subset.T


def farthest_point_sampling(points, num_samples, metric: float = 2):
    num_points = points.shape[0]
    distances = torch.full((num_points, 1), torch.inf)
    samples = torch.zeros(num_samples)
    farthest_idx = torch.randint(0, num_points, (1,)).item()

    for i in tqdm(range(num_samples)):
        samples[i] = farthest_idx
        current_point = points[farthest_idx]
        distances_to_current_point = torch.cdist(
            points, current_point.unsqueeze(0), p=metric
        )
        distances = torch.tensor(
            [torch.min(x, y) for x, y in zip(distances, distances_to_current_point)]
        )
        farthest_idx = torch.argmax(distances).item()

    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="find correlated ReLUs")
    parser.add_argument(
        "activations",
        help="input activations",
        default="cached_relus/cifar100_with_val/resnet18/activations.pkl",
    )
    parser.add_argument(
        "num_prototypes",
        type=int,
        help="the number of prototypes to use.",
        default=1000,
    )
    parser.add_argument(
        "outdir",
        help="cache output directory",
        default="cached_relus/cifar100_with_val/resnet18/100_most_important_relus",
    )
    return parser.parse_args()


def associate_relus_to_prototypes(activations, prototypes, outdir):

    layer_1_0_relu_1 = activations["layer1[0].relu_1"]
    D, H, W = layer_1_0_relu_1.shape[1:]
    layer_1_0_relu_1 = layer_1_0_relu_1.flatten(1).T
    print(
        f"Now for each ReLU, let's calculate the prototype it is most "
        f"correlated with.."
    )
    print(f"Each prototype and ReLU is with dimension: {prototypes.shape[-1]}")
    print(
        f"Associating {layer_1_0_relu_1.shape[0]} ReLUs with"
        f" {prototypes.shape[0]} prototypes..."
    )

    for layer_name in [
        "layer1[0].relu_1",
        "layer1[0].relu_2",
        "layer1[1].relu_1",
        "layer1[1].relu_2",
        "layer2[0].relu_1",
        "layer2[0].relu_2",
        "layer2[1].relu_1",
        "layer2[1].relu_2",
        "layer3[0].relu_1",
        "layer3[0].relu_2",
        "layer3[1].relu_1",
        "layer3[1].relu_2",
        "layer4[0].relu_1",
        "layer4[0].relu_2",
        "layer4[1].relu_1",
        "layer4[1].relu_2",
    ]:
        start_time = time.time()
        layer_responses = activations[layer_name].flatten(1).T
        _chunk_size = 1000
        closest_prototype_chunks = []
        for i in tqdm(range(layer_responses.shape[0] // _chunk_size + 1)):
            closest_prototype_chunks.append(
                torch.from_numpy(
                    metrics.pairwise_distances_argmin(
                        layer_responses[i * _chunk_size : (i + 1) * _chunk_size],
                        prototypes.T,
                        metric="hamming",
                    )
                )
            )
        end_time = time.time()
        print(
            f"associating ReLUs with prototypes for layer {layer_name} took"
            f" {end_time - start_time} seconds"
        )
        closest_prototypes = closest_prototype_chunks
        with open(osp.join(outdir, f"closest_prototypes_{layer_name}.pkl"), "wb") as f:
            pickle.dump(closest_prototypes, f)


def main():
    args = parse_args()
    # outdir = "cached_relus/cifar100_with_val/resnet18/100_most_important_relus"
    os.makedirs(args.outdir, exist_ok=True)
    activations, prototypes = extract_top_k_most_frequently_associated_prototypes(
        args.num_prototypes
    )
    associate_relus_to_prototypes(activations, prototypes, args.outdir)


if __name__ == "__main__":
    main()
