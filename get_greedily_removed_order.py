import os
import json
import argparse
import pickle

ROOT = "results/cifar100_with_val/resnet18/100_most_important_prototypes"

DEFAULT_LAYER_NAME_TO_CHOOSING_MATRIX_PATH = (
    "cached_relus/cifar100_with_val/resnet18/"
    "100_most_important_relus/"
    "layer_name_to_choosing_matrix/layer_name_to_matrix.pkl"
)

PATH_TO_CONFIG_ROOTS = (
    "research/configs/classification/resnet"
    "/resnet18_cifar100_resplit_with_val_100_prototypes"
)
GENERATED_CHECKPOINTS_DEFAULT_PATH = (
    "osi_checkpoints/cifar100/resnet18_cifar100_resplit_with_val_100_prototypes"
)
FINETUNED_MODELS_ROOT_PATH = (
    "trained_networks/classification/resnet18_cifar100_resplit_with_val_100_prototypes"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="fix layer names according to grid search results in the "
        "config files"
    )
    parser.add_argument(
        "results_root", help="path to greedy search algorithm results.", default=ROOT
    )
    parser.add_argument(
        "layer_name_to_choosing_matrix_path",
        help="path to the pickle holding the mapping between layer name and "
        "choosing matrix.",
        default=DEFAULT_LAYER_NAME_TO_CHOOSING_MATRIX_PATH,
    )
    parser.add_argument(
        "configs_root",
        help="path to configs root directory",
        default=PATH_TO_CONFIG_ROOTS,
    )
    parser.add_argument(
        "generated_checkpoints_path",
        help="path to the generated checkpoints root path",
        default=GENERATED_CHECKPOINTS_DEFAULT_PATH,
    )
    parser.add_argument(
        "finetuned_models_path",
        help="path to the root holding finetuned models",
        default=FINETUNED_MODELS_ROOT_PATH,
    )
    return parser.parse_args()


def get_removed_layers_sequence(args):
    d = json.load(
        open(
            os.path.join(
                args.results_root,
                "all_induced_relus_except_from_one_layer_name_to_acc_added_1.txt",
            ),
            "r",
        )
    )
    removed = []
    for i in range(2, 16 + 1):
        new_d = json.load(
            open(
                os.path.join(
                    args.results_root,
                    f"all_induced_relus_except_from_one_layer_name_to_acc_added_"
                    f"{i}.txt",
                ),
                "r",
            )
        )
        not_in_curr = set(d.keys()) ^ set(new_d.keys())
        removed += [x for x in not_in_curr if x not in removed]
    removed += list(new_d.keys())
    print(removed)
    return removed


def build_before_training_config_file_text(args, num_replacements=3):
    header = f"""import pickle
import os.path as osp

NUM_OF_REPLACEMENTS = {num_replacements}


_base_ = [
    '../../_base_/datasets/cifar100_bs64.py',
    '../../_base_/schedules/cifar10_bs128.py',
]"""

    new_order = f"\n\nremoved_sequence = [ \n"
    removed = get_removed_layers_sequence(args)
    for layer in removed:
        new_order += f"    '{layer}', \n"
    new_order += "] \n"

    body = f"""def load_resnet18_choosing_matrices():
    with open(
        osp.join(
            '{args.layer_name_to_choosing_matrix_path}'
        ),
        "rb",
    ) as f:
        layer_name_to_matrix = pickle.load(f)
    return layer_name_to_matrix"""
    footer = """
def layer_name_to_row_col(layer_name):
    row = int(layer_name.split("layer")[-1].split("[")[0]) - 1
    level = int(layer_name.split("[")[-1].split("]")[0])
    sub = int(layer_name.split("_")[-1])
    col = 2 * int(level) + sub - 1
    return row, col


USE_INDUCED_RELU = [
    [False] * 2 * 2
    for layer in [
        2,
    ]
    * 4
]
for r in removed_sequence[:NUM_OF_REPLACEMENTS]:
    row, col = layer_name_to_row_col(r)
    USE_INDUCED_RELU[row][col] = True


def load_permutation_matrices_for_resnet18():
    layer_name_to_matrix = load_resnet18_choosing_matrices()
    permutation_matrices = []
    for layer_index in [1, 2, 3, 4]:
        for sub_index in [0, 1]:
            for act_index in [1, 2]:
                layer_name = f"layer{layer_index}[{sub_index}].relu_{act_index}"
                permutation_matrices.append(layer_name_to_matrix[layer_name].cuda())
    return permutation_matrices


# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        # type='ResNet_CIFAR',
        # depth=18,
        # num_stages=4,
        # out_indices=(3, ),
        # style='pytorch',
        type="ResNet_InducedReLU",
        num_classes=100,
        is_cifar=True,
        permutation_matrices=load_permutation_matrices_for_resnet18(),
        use_induced_relu=USE_INDUCED_RELU,
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=100,
        in_channels=512,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)

optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.001)
lr_config = dict(policy="step", step=[60, 120, 160], gamma=0.2)

# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable

dist_params = dict(backend="gloo")
log_level = "INFO"
load_from = None
resume_from = None
evaluation = dict(interval=10, by_epoch=True)
workflow = [("train", 10), ("val", 1)]

"""
    content = header + new_order + body + footer
    return content


def build_for_training_config_file_text(args, num_replacements):
    header = f"""import os
import pickle
import os.path as osp

NUM_OF_REPLACEMENTS = {num_replacements}

_base_ = [
    "../../_base_/datasets/cifar100_bs64.py",
    "../../_base_/schedules/cifar10_bs128.py",
]


def layer_name_to_row_col(layer_name):
    row = int(layer_name.split("layer")[-1].split("[")[0]) - 1
    level = int(layer_name.split("[")[-1].split("]")[0])
    sub = int(layer_name.split("_")[-1])
    col = 2 * int(level) + sub - 1
    return row, col
"""
    new_order = f"\n\nremoved_sequence = [ \n"
    removed = get_removed_layers_sequence(args)
    for layer in removed:
        new_order += f"    '{layer}', \n"
    new_order += "] \n"

    body1 = f"""\nDEFAULT_PICLE_PATH = '{args.layer_name_to_choosing_matrix_path}'
    \n"""
    body2 = """
USE_INDUCED_RELU = [
    [False] * 2 * 2
    for layer in [
        2,
    ]
    * 4
]
for r in removed_sequence[:NUM_OF_REPLACEMENTS]:
    row, col = layer_name_to_row_col(r)
    print(row, col)
    USE_INDUCED_RELU[row][col] = True


def load_resnet18_choosing_matrices(
    layer_name_to_matrix_pickle_path=DEFAULT_PICLE_PATH,
):
    with open(layer_name_to_matrix_pickle_path, "rb") as f:
        layer_name_to_matrix = pickle.load(f)
    return layer_name_to_matrix


def load_permutation_matrices_for_resnet18():
    layer_name_to_matrix = load_resnet18_choosing_matrices()
    permutation_matrices = []
    for layer_index in [1, 2, 3, 4]:
        for sub_index in [0, 1]:
            for act_index in [1, 2]:
                layer_name = f"layer{layer_index}[{sub_index}].relu_{act_index}"
                permutation_matrices.append(layer_name_to_matrix[layer_name])
    return permutation_matrices


# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        # type='ResNet_CIFAR',
        # depth=18,
        # num_stages=4,
        # out_indices=(3, ),
        # style='pytorch',
        type="ResNet_InducedReLU",
        num_classes=100,
        is_cifar=True,
        permutation_matrices=load_permutation_matrices_for_resnet18(),
        use_induced_relu=USE_INDUCED_RELU,
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=100,
        in_channels=512,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)

optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.001)
lr_config = dict(policy="step", step=[60, 120, 160], gamma=0.2)

# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable

dist_params = dict(backend="gloo")
log_level = "INFO"
load_from = None
"""
    footer = (
        f'resume_from = os.path.join("{args.generated_checkpoints_path}",\n'
        '    f"replaced_{NUM_OF_REPLACEMENTS}_relus_greedy/",\n'
        '    f"greedy_search_for_{NUM_OF_REPLACEMENTS}_induced_relus.pth") '
        "\n"
        + """evaluation = dict(interval=10, by_epoch=True)
workflow = [("train", 10), ("val", 1), ("val", 1)]
"""
    )
    return header + new_order + body1 + body2 + footer


def build_after_training_config_file(args, num_replacements):
    header = f"""import pickle
import os.path as osp


NUM_OF_REPLACEMENTS = {num_replacements}


_base_ = [
    "../../_base_/datasets/cifar100_bs64.py",
    "../../_base_/schedules/cifar10_bs128.py",
]
"""
    new_order = f"\n\nremoved_sequence = [ \n"
    removed = get_removed_layers_sequence(args)
    for layer in removed:
        new_order += f"    '{layer}', \n"
    new_order += "] \n"
    body1 = f"""
def layer_name_to_row_col(layer_name):
    row = int(layer_name.split("layer")[-1].split("[")[0]) - 1
    level = int(layer_name.split("[")[-1].split("]")[0])
    sub = int(layer_name.split("_")[-1])
    col = 2 * int(level) + sub - 1
    return row, col


USE_INDUCED_RELU = [
    [False] * 2 * 2
    for layer in [
        2,
    ]
    * 4
]
for r in removed_sequence[:NUM_OF_REPLACEMENTS]:
    row, col = layer_name_to_row_col(r)
    USE_INDUCED_RELU[row][col] = True
    
DEFAULT_PICLE_PATH = "{args.layer_name_to_choosing_matrix_path}"
"""
    body2 = """
def load_resnet18_choosing_matrices(
    layer_name_to_matrix_pickle_path=DEFAULT_PICLE_PATH,
):
    with open(layer_name_to_matrix_pickle_path, "rb") as f:
        layer_name_to_matrix = pickle.load(f)
    return layer_name_to_matrix


def load_permutation_matrices_for_resnet18():
    layer_name_to_matrix = load_resnet18_choosing_matrices()
    permutation_matrices = []
    for layer_index in [1, 2, 3, 4]:
        for sub_index in [0, 1]:
            for act_index in [1, 2]:
                layer_name = f"layer{layer_index}[{sub_index}].relu_{act_index}"
                permutation_matrices.append(layer_name_to_matrix[layer_name])
    return permutation_matrices


# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        # type='ResNet_CIFAR',
        # depth=18,
        # num_stages=4,
        # out_indices=(3, ),
        # style='pytorch',
        type="ResNet_InducedReLU",
        num_classes=100,
        is_cifar=True,
        permutation_matrices=load_permutation_matrices_for_resnet18(),
        use_induced_relu=USE_INDUCED_RELU,
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=100,
        in_channels=512,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)

optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.001)
lr_config = dict(policy="step", step=[60, 120, 160], gamma=0.2)

# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable

dist_params = dict(backend="gloo")
log_level = "INFO"
load_from = None
# TODO: write a script to create a checkpoint!"
"""
    footer1 = (
        'resume_from = osp.join("'
        f"{args.finetuned_models_path}"
        + '", \n     '
        + 'f"/induced_relu_backbone_greedly_swap_{'
        'NUM_OF_REPLACEMENTS}_relus_finetune/latest.pth") \n'
    )
    footer2 = (
        'checkpoint = osp.join("'
        f"{args.finetuned_models_path}"
        + '", \n     '
        + 'f"/induced_relu_backbone_greedly_swap_{'
        'NUM_OF_REPLACEMENTS}_relus_finetune/latest.pth") \n'
    )
    footer3 = """evaluation = dict(interval=10, by_epoch=True)
workflow = [("train", 10), ("val", 1), ("val", 1)]
    """
    return header + new_order + body1 + body2 + footer1 + footer2 + footer3


def main():
    args = parse_args()
    removed_sequence = get_removed_layers_sequence(args)
    with open(os.path.join(args.configs_root, "removed_sequence.pkl"), "wb") as f:
        pickle.dump(removed_sequence, f)
    for i in range(0, 16 + 1):
        config_file_content = build_before_training_config_file_text(args, i)
        with open(
            os.path.join(args.configs_root, f"induced_relu_" f"{i}_induced.py"), "w"
        ) as f:
            f.write(config_file_content)
        config_file_content = build_for_training_config_file_text(args, i)
        with open(
            os.path.join(
                args.configs_root,
                f"induced_relu_finetune_network_for_greedly_swaped_{i}_relu.py",
            ),
            "w",
        ) as f:
            f.write(config_file_content)
        config_file_content = build_after_training_config_file(args, i)
        with open(
            os.path.join(
                args.configs_root,
                f"induced_relu_test_network_for_greedly_swaped_"
                f"{i}_relu_after_finetuning.py",
            ),
            "w",
        ) as f:
            f.write(config_file_content)
        print()


if __name__ == "__main__":
    main()
