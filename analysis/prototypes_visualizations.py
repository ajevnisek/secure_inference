import os
import torch
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import unravel_index

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

os.makedirs('results/analysis/', exist_ok=True)
path = 'cached_relus/cifar100_with_val/resnet18/closest_prototypes_layer1[0].relu_1.pkl'


with open(path, 'rb') as f:
    closest = pickle.load(f)

prototypes_indices_path = 'cached_relus/cifar100_with_val/resnet18/1K_prototypes_indices_p=0.pkl'
with open(prototypes_indices_path, 'rb') as f:
    prototypes_indices = pickle.load(f)

closest = torch.cat(closest)
prototypes_locations = unravel_index(prototypes_indices.int(), (32, 32, 64))
X, Y, Z = prototypes_locations

plt.clf()
sns.jointplot(data={'row': prototypes_locations[0],
                    'col': prototypes_locations[1],
                    'channel': prototypes_locations[2]},
              x='col', y='row',hue='channel')
plt.xlim([0, 32])
plt.ylim([0, 32])
plt.savefig('results/analysis/prototypes_visualization.png')
plt.clf()
sns.jointplot(data={'row': prototypes_locations[0],
                    'col': prototypes_locations[1],
                    'channel': prototypes_locations[2]},
              x='col', y='row')
plt.xlim([0, 32])
plt.ylim([0, 32])
plt.savefig('results/analysis/prototypes_visualization_no_hue.png')

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z)
ax.set_xlabel('row')
ax.set_ylabel('col')
ax.set_zlabel('channel')
ax.set_title('1K prototypes locations in first activation layer:\n Farthest Point Sampling Result')
def animate(i):
    ax.view_init(elev=30, azim=45+i, roll=15)
x = np.linspace(0, 100)
ani = animation.FuncAnimation(fig, animate, repeat=True,
                                    frames=len(x) - 1, interval=100)
writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save('results/analysis/prototypes.gif', writer=writer)
plt.clf()
for layer_name in layer_names:
    path = f'cached_relus/cifar100_with_val/resnet18/' \
           f'closest_prototypes_{layer_name}.pkl'
    with open(path, 'rb') as f:
        closest_indices = pickle.load(f)
    path = 'cached_relus/cifar100_with_val/resnet18/1K_prototypes_indices_p=0.pkl'
    with open(path, 'rb') as f:
        prototypes = pickle.load(f)
    closest_indices = torch.cat(closest_indices)
    curr_layer_prototypes = torch.index_select(prototypes.T, 0, closest_indices.T).T

    deep_layer_where_is_the_prototype_location = unravel_index(
        curr_layer_prototypes.int(), (32, 32, 64))
    X, Y, Z = deep_layer_where_is_the_prototype_location

    plt.clf()
    sns.jointplot(data={'row': X,
                        'col': Y,
                        'channel': Z},
                  x='col', y='row',hue='channel')
    plt.xlim([0, 32])
    plt.ylim([0, 32])
    plt.legend(loc='best', ncol=3, title='scatter')
    plt.suptitle(f"Prototypes used in layer: {layer_name}")
    plt.tight_layout()
    plt.savefig(f'results/analysis/which_prototypes_are_used_in_'
                f'{layer_name}.png')

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    from collections import Counter
    histogram = Counter([(x.item(), y.item(), z.item()) for x, y, z in zip(X, Y, Z)] )
    ax.scatter(np.array(list(histogram.keys()))[... ,0],
               np.array(list(histogram.keys()))[... ,1],
               np.array(list(histogram.keys()))[..., 2], s=list(histogram.values()))
    ax.set_xlabel('row')
    ax.set_ylabel('col')
    ax.set_zlabel('channel')
    def animate(i):
        ax.view_init(elev=30, azim=45 + i, roll=15)


    title = f'1K prototypes locations in first activation layer:\n Size ' \
            f'encodes usage in layer: {layer_name}'
    ax.set_title(title)
    x = np.linspace(0, 100)
    ani = animation.FuncAnimation(fig, animate, repeat=True,
                                  frames=len(x) - 1, interval=100)
    writer = animation.PillowWriter(fps=15,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    ani.save('results/analysis/' + title.replace(' ', '_') + '.gif',
             writer=writer)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    from collections import Counter
    import matplotlib.cm as cm

    histogram = Counter([(x.item(), y.item(), z.item()) for x, y, z in zip(X, Y, Z)] )
    ax.scatter(np.array(list(histogram.keys()))[... ,0],
               np.array(list(histogram.keys()))[... ,1],
               np.array(list(histogram.keys()))[..., 2], c=list(
            histogram.values()), cmap=cm.jet)
    ax.set_xlabel('row')
    ax.set_ylabel('col')
    ax.set_zlabel('channel')
    def animate(i):
        ax.view_init(elev=30, azim=45 + i, roll=15)


    title = f'1K prototypes locations in first activation layer:\n Size ' \
            f'encodes usage in layer: {layer_name}'
    ax.set_title(title)
    x = np.linspace(0, 100)
    ani = animation.FuncAnimation(fig, animate, repeat=True,
                                  frames=len(x) - 1, interval=100)
    writer = animation.PillowWriter(fps=15,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    ani.save('results/analysis/cm_' + title.replace(' ', '_') + '.gif',
             writer=writer)


layer_name_to_histogram = {}
for layer_name in layer_names:
    path = f'cached_relus/cifar100_with_val/resnet18/' \
           f'closest_prototypes_{layer_name}.pkl'
    with open(path, 'rb') as f:
        closest_indices = pickle.load(f)
    path = 'cached_relus/cifar100_with_val/resnet18/1K_prototypes_indices_p=0.pkl'
    with open(path, 'rb') as f:
        prototypes = pickle.load(f)
    closest_indices = torch.cat(closest_indices)
    curr_layer_prototypes = torch.index_select(prototypes.T, 0, closest_indices.T).T

    deep_layer_where_is_the_prototype_location = unravel_index(
        curr_layer_prototypes.int(), (32, 32 , 64))
    X, Y, Z = deep_layer_where_is_the_prototype_location
    from collections import Counter
    histogram = Counter(
        [(x.item(), y.item(), z.item()) for x, y, z in zip(X, Y, Z)])
    layer_name_to_histogram[layer_name] = histogram

all_counters = [c for c in layer_name_to_histogram.values()]
all_stats = all_counters[0]
for c in all_counters[1:]:
    all_stats += c

prots = list(all_stats.keys())
prots.sort(key=lambda x: all_stats[x])
usage = list(all_stats.values())
usage.sort()

plt.clf()
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(usage) / np.sum(usage))
plt.grid(True)
plt.xlabel('prototype dummy index')
plt.ylabel('CDF')
plt.title('CDF usage for prototype')
plt.subplot(1, 2, 2)
plt.plot(range(1000-10 + 1, 1000 + 1), (np.cumsum(usage) / np.sum(usage))[-10:], '-x')
plt.grid(True)
plt.xlabel('prototype: (row, col, channel)')
plt.ylabel('CDF')
plt.title('zoom in on 10 most recurring prototypes')
plt.xticks(range(1000-10 + 1, 1000 + 1), prots[-10:], rotation=270)
plt.tight_layout()
fig=plt.gcf()
fig.set_size_inches((13, 4))
plt.tight_layout()
plt.savefig('results/analysis/prototypes_usage_cdf.png')



with open('cached_relus/cifar100_with_val/resnet18/'
          '1K_protorypes_usage_hist.pkl',
          'wb') as f:
    pickle.dump(all_stats, f)



np.ravel_multi_index(prots[0], (32, 32, 64)) in prototypes
with open('cached_relus/cifar100_with_val/1K_protorypes_usage_hist.pkl',
          'wb') as f:
    pickle.dump(all_stats, f)


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
outdir = 'cached_relus/cifar100_with_val/resnet18/100_most_important_relus'
os.makedirs(outdir, exist_ok=True)
with open('cached_relus/cifar100_with_val/resnet18/activations.pkl', 'rb') as f:
    activations = pickle.load(f)
import os.path as osp
with open(osp.join('cached_relus/cifar100_with_val/resnet18/',
                       '1K_prototypes_p=0.pkl'), 'rb') as f:
    prototypes_values = pickle.load(f)

with open('cached_relus/cifar100_with_val/resnet18/'
          '1K_protorypes_usage_hist.pkl', 'rb') as f:
    all_stats = pickle.load(f)

prots = list(all_stats.keys())
prots.sort(key=lambda x: all_stats[x])
usage = list(all_stats.values())
usage.sort()
prototypes_indices_path = 'cached_relus/cifar100_with_val/resnet18/1K_prototypes_indices_p=0.pkl'
with open(prototypes_indices_path, 'rb') as f:
    prototypes_indices = pickle.load(f)

get_index = lambda tensor, x:  ((tensor == x).nonzero(as_tuple=True)[0])
prototypes_subset_indices = np.ravel_multi_index(
    ([x[0] for x in prots[:100]],
     [x[1] for x in prots[:100]],
     [x[2] for x in prots[:100]]),
    (32, 32, 64))
locations_of_selected_prototypes = [get_index(prototypes_indices, x)
                                    for x in prototypes_subset_indices]
prots_subset = torch.index_select(prototypes_values.T, 0,
                                  torch.tensor(
                                      locations_of_selected_prototypes))


import time
from tqdm import tqdm
import sklearn.metrics as metrics
associate_relus_to_prototypes(activations, prots_subset.T, outdir)
