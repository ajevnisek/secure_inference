import os.path

import torch
import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import hamming_loss


path_activations = 'cached_relus/cifar100_with_val/resnet18/activations.pkl'
with open(path_activations, 'rb') as f:
    activations = pickle.load(f)
for k in activations:
        print(k, activations[k].shape)
path = 'cached_relus/cifar100_with_val/resnet18/1K_prototypes_p=0.pkl'
with open(path, 'rb') as f:
    prototypes = pickle.load(f)
layer_names = [
        'layer1[0].relu_1', 'layer1[0].relu_2',
        'layer1[1].relu_1', 'layer1[1].relu_2',
        'layer2[0].relu_1', 'layer2[0].relu_2',
        'layer2[1].relu_1', 'layer2[1].relu_2',
        'layer3[0].relu_1', 'layer3[0].relu_2',
        'layer3[1].relu_1', 'layer3[1].relu_2',
        'layer4[0].relu_1', 'layer4[0].relu_2',
        'layer4[1].relu_1', 'layer4[1].relu_2',
    ]

layer_name_to_stats = {}
for layer_name in layer_names:
    path_indices = f'cached_relus/cifar100_with_val/resnet18/' \
                   f'closest_prototypes_{layer_name}.pkl'
    with open(path_indices, 'rb') as f:
        prototypes_indices = pickle.load(f)
    prototypes_indices = torch.cat(prototypes_indices)
    curr_layer_prototypes = torch.index_select(prototypes.T, 0, prototypes_indices.int()).T

    losses = []
    for act, prot in tqdm(zip(activations[layer_name].flatten(1).T,
                              curr_layer_prototypes.T)):
        losses.append(hamming_loss(act, prot))
    losses = torch.tensor(losses)
    print(layer_name, losses.mean().item(), losses.std().item())
    layer_name_to_stats[layer_name] = {'mean': losses.mean().item(),
                                       'std': losses.std().item()}
plt.clf()
plt.bar(range(len(layer_names)), [layer_name_to_stats[layer_name]['mean']*100.0
                                  for layer_name in layer_names])
plt.bar(range(len(layer_names)), [layer_name_to_stats[layer_name]['std'] * 100.0
                                  for layer_name in layer_names])
plt.legend(['mean', 'std'])
plt.xticks(range(len(layer_names)), layer_names, rotation=90)
plt.grid(True)
title = 'Hamming Distance (% from vector length) ' \
        'between prototype response and DReLU response'
plt.title()
plt.xlabel('layer name (ordered)')
plt.ylabel('Hamming Distance')
plt.tight_layout()
plt.savefig(os.path.join('results/analysis/',
                         title.replace(' ', '_') + '.png'))
