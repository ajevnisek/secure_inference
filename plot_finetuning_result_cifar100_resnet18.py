import argparse
import os.path as osp
import matplotlib.pyplot as plt

# ROOT = "results/test_logs/resnet18_cifar100_resplit_with_val_1000_prototypes"
# REMOVED_LAYERS_TEXT_FILES_ROOT = (
#     "results/cifar100_with_val/resnet18/1000_most_important_prototypes"
# )
# OUTPUT_DIR = "results/cifar100_with_val/resnet18/1000_most_important_prototypes"
parser = argparse.ArgumentParser(description="run visualization")
parser.add_argument("num_prototypes", type=int, help="num of prototypes", default=1000)
args = parser.parse_args()

ROOT = osp.join(
    f"results/test_logs/",
    f"resnet18_cifar100_resplit_with_val_{args.num_prototypes}_prototypes",
)
REMOVED_LAYERS_TEXT_FILES_ROOT = osp.join(
    "results/cifar100_with_val/resnet18/",
    f"{args.num_prototypes}_most_important_prototypes",
)
OUTPUT_DIR = osp.join(
    "results/cifar100_with_val/resnet18",
    f"{args.num_prototypes}_most_important_prototypes",
)


def get_data(path):
    with open(path, "r") as f:
        data = f.read()
    acc1 = float(
        [l for l in data.splitlines() if l.startswith("accuracy_top-1 : ")][-1].split(
            "accuracy_top-1 : "
        )[-1]
    )
    acc5 = float(
        [l for l in data.splitlines() if l.startswith("accuracy_top-5 : ")][-1].split(
            "accuracy_top-5 : "
        )[-1]
    )
    return {"acc1": acc1, "acc5": acc5}


get_after_finetuning_path = lambda x: osp.join(
    ROOT, "after_finetune", f"after_finetune_{x}.log"
)
get_before_finetuning_path = lambda x: osp.join(
    ROOT, "before_finetune", f"before_finetune_{x}.log"
)

after_finetuning_acc1 = []
after_finetuning_acc5 = []
before_finetuning_acc1 = []
before_finetuning_acc5 = []

for i in range(1, 16 + 1):
    path = get_after_finetuning_path(i)
    result = get_data(path)
    after_finetuning_acc1.append(result["acc1"])
    after_finetuning_acc5.append(result["acc5"])
    path = get_before_finetuning_path(i)
    result = get_data(path)
    before_finetuning_acc1.append(result["acc1"])
    before_finetuning_acc5.append(result["acc5"])

original_acc = 75.06
import json

json.load(
    open(
        osp.join(
            REMOVED_LAYERS_TEXT_FILES_ROOT,
            "all_induced_relus_except_from_one_layer_name_to_acc_added_1.txt",
        ),
        "r",
    )
)
d = json.load(
    open(
        osp.join(
            REMOVED_LAYERS_TEXT_FILES_ROOT,
            "all_induced_relus_except_from_one_layer_name_to_acc_added_1.txt",
        ),
        "r",
    )
)
removed = []
for i in range(2, 16 + 1):
    new_d = json.load(
        open(
            osp.join(
                REMOVED_LAYERS_TEXT_FILES_ROOT,
                f"all_induced_relus_except_from_one_layer_name_to_acc_added_"
                f"{i}.txt",
            ),
            "r",
        )
    )
    not_in_curr = set(d.keys()) ^ set(new_d.keys())
    removed += [x for x in not_in_curr if x not in removed]
print(removed)
len(removed)
all_layers = [
    f"layer{i}[{j}].relu_{k}"
    for i in range(1, 1 + 4)
    for j in range(0, 2)
    for k in range(1, 2 + 1)
]
removed += list(new_d.keys())
removed_sequence = removed


# def draw_arrow(plt, arr_start, arr_end):
#     dx = arr_end[0] - arr_start[0]
#     dy = arr_end[1] - arr_start[1]
#     plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=0.2,
#               head_length=3, length_includes_head=True, color='black')
plt.plot(range(1, 16 + 1), [original_acc] * 16, "--k", linewidth=3)
plt.plot(range(1, 16 + 1), before_finetuning_acc1, "-x", linewidth=3)
plt.plot(range(1, 16 + 1), after_finetuning_acc1, "-x", linewidth=3)
# draw_arrow(plt, (9, before_finetuning_acc1[8]), (9, after_finetuning_acc1[8]))
plt.gca().annotate(
    "",
    xy=(9, after_finetuning_acc1[8]),
    xytext=(9, before_finetuning_acc1[8]),
    arrowprops=dict(arrowstyle="->"),
)
plt.gca().text(
    9,
    (before_finetuning_acc1[8] + after_finetuning_acc1[8]) / 2.0,
    f"+{after_finetuning_acc1[8] - before_finetuning_acc1[8]} [%]",
)
plt.xticks(range(1, 1 + 16), removed_sequence, rotation=90)
plt.xlabel("iteration step")
plt.ylabel("accuracy [%]")
plt.title("greedy search of ReLU to Induced ReLU swaps")
plt.grid(True)
plt.legend(["baseline", "greedy", "finetune"])
plt.tight_layout()
plt.savefig(osp.join(OUTPUT_DIR, "greedy_search_of_induced_relus_with_finetune.png"))

plt.clf()
plt.plot(range(1, 16 + 1), [original_acc] * 16, "--k", linewidth=3)
plt.plot(range(1, 16 + 1), before_finetuning_acc1, "-x", linewidth=3)
plt.plot(range(1, 16 + 1), after_finetuning_acc1, "-x", linewidth=3)
# draw_arrow(plt, (9, before_finetuning_acc1[8]), (9, after_finetuning_acc1[8]))
for i in range(4, 16, 2):
    plt.gca().annotate(
        "",
        xy=(i + 1, after_finetuning_acc1[i]),
        xytext=(i + 1, before_finetuning_acc1[i]),
        arrowprops=dict(arrowstyle="->"),
    )
    plt.gca().text(
        i + 1,
        (before_finetuning_acc1[i] + after_finetuning_acc1[i]) / 2.0,
        f"x" f"{after_finetuning_acc1[i] / before_finetuning_acc1[i]:.0f}",
    )
plt.xticks(range(1, 1 + 16), removed_sequence, rotation=90)
plt.xlabel("iteration step")
plt.ylabel("accuracy [%]")
plt.title("greedy search of ReLU to Induced ReLU swaps")
plt.grid(True)
plt.legend(["baseline", "greedy", "finetune"])
plt.tight_layout()
plt.savefig(
    osp.join(OUTPUT_DIR, "greedy_search_of_induced_relus_with_finetune_mul.png")
)
from collections import OrderedDict

layer_name_to_num_of_relus = OrderedDict(
    {
        "act1": 65536,
        "layer1[0].relu_1": 65536,
        "layer1[0].relu_2": 65536,
        "layer1[1].relu_1": 65536,
        "layer1[1].relu_2": 65536,
        "layer2[0].relu_1": 32768,
        "layer2[0].relu_2": 32768,
        "layer2[1].relu_1": 32768,
        "layer2[1].relu_2": 32768,
        "layer3[0].relu_1": 16384,
        "layer3[0].relu_2": 16384,
        "layer3[1].relu_1": 16384,
        "layer3[1].relu_2": 16384,
        "layer4[0].relu_1": 8192,
        "layer4[0].relu_2": 8192,
        "layer4[1].relu_1": 8192,
        "layer4[1].relu_2": 8192,
    }
)

relu_usage = []
for i in range(len(removed_sequence)):
    relus_reduced = 0
    for layer in removed_sequence[: i + 1]:
        relus_reduced += layer_name_to_num_of_relus[layer]
    relu_usage.append(
        100.0
        * (sum(layer_name_to_num_of_relus.values()) - relus_reduced)
        / sum(layer_name_to_num_of_relus.values())
    )

plt.clf()
plt.plot(range(1, 16 + 1), [original_acc] * 16, "--k", linewidth=3)
plt.plot(range(1, 16 + 1), before_finetuning_acc1, "-x", linewidth=3)
plt.plot(range(1, 16 + 1), after_finetuning_acc1, "-x", linewidth=3)
plt.gca().annotate(
    "",
    xy=(9, after_finetuning_acc1[8]),
    xytext=(9, before_finetuning_acc1[8]),
    arrowprops=dict(arrowstyle="->"),
)
plt.gca().text(
    9,
    (before_finetuning_acc1[8] + after_finetuning_acc1[8]) / 2.0,
    f"+{after_finetuning_acc1[8] - before_finetuning_acc1[8]} [%]",
)
plt.xticks(range(1, 1 + 16), [f"{x:.1f}" for x in relu_usage], rotation=0)
plt.xlabel("ReLU usage [%]")
plt.ylabel("accuracy [%]")
plt.title("CIFAR100 ResNet18: Accuracy vs ReLU usage")
plt.grid(True)
plt.legend(["baseline", "greedy", "finetune"])
fig = plt.gcf()
fig.set_size_inches((10, 5))
plt.tight_layout()
plt.savefig(
    osp.join(
        OUTPUT_DIR, "greedy_search_of_induced_relus_with_finetune_x_axis_relu_usage.png"
    )
)


plt.clf()
plt.plot(range(1, 16 + 1), [original_acc] * 16, "--k", linewidth=3)
plt.plot(range(1, 16 + 1), before_finetuning_acc1, "-x", linewidth=3)
plt.plot(range(1, 16 + 1), after_finetuning_acc1, "-x", linewidth=3)
# draw_arrow(plt, (9, before_finetuning_acc1[8]), (9, after_finetuning_acc1[8]))
for i in range(4, 16, 2):
    plt.gca().annotate(
        "",
        xy=(i + 1, after_finetuning_acc1[i]),
        xytext=(i + 1, before_finetuning_acc1[i]),
        arrowprops=dict(arrowstyle="->"),
    )
    plt.gca().text(
        i + 1,
        (before_finetuning_acc1[i] + after_finetuning_acc1[i]) / 2.0,
        f"x" f"{after_finetuning_acc1[i] / before_finetuning_acc1[i]:.0f}",
    )
plt.xticks(range(1, 1 + 16), [f"{x:.1f}" for x in relu_usage], rotation=0)
plt.xlabel("ReLU usage [%]")
plt.ylabel("accuracy [%]")
plt.title("CIFAR100 ResNet18: Accuracy vs ReLU usage")
plt.grid(True)
plt.legend(["baseline", "greedy", "finetune"])
fig = plt.gcf()
fig.set_size_inches((10, 5))
plt.tight_layout()
plt.savefig(
    osp.join(
        OUTPUT_DIR,
        "greedy_search_of_induced_relus_with_finetune_mul_x_axis_relu_usage.png",
    )
)
plt.show()
