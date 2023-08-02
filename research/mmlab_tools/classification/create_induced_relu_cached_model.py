# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import argparse
import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (
    auto_select_device,
    get_root_logger,
    setup_multi_processes,
    wrap_distributed_model,
    wrap_non_distributed_model,
)

from research.distortion.arch_utils.factory import arch_utils_factory
import pickle

# from research.mmlab_extension.classification.resnet_cifar_v2 import ResNet_CIFAR_V2  # TODO: why is this needed?
from research.mmlab_extension.classification.resnet import (
    MyResNet,
)  # TODO: why is this needed?


REMOVED_SEQUENCE = (
    "layer4[1].relu_2",
    "layer1[0].relu_1",
    "layer1[1].relu_2",
    "layer2[0].relu_2",
    "layer1[1].relu_1",
    "layer1[0].relu_2",
    "layer2[1].relu_1",
    "layer3[1].relu_1",
    "layer4[0].relu_1",
    "layer3[1].relu_2",
    "layer3[0].relu_2",
    "layer3[0].relu_1",
    "layer2[1].relu_2",
    "layer2[0].relu_1",
    "layer4[1].relu_1",
    "layer4[0].relu_2",
)


def layer_name_to_row_col(layer_name):
    row = int(layer_name.split("layer")[-1].split("[")[0]) - 1
    level = int(layer_name.split("[")[-1].split("]")[0])
    sub = int(layer_name.split("_")[-1])
    col = 2 * int(level) + sub - 1
    return row, col


def create_use_induced_relu_bit_vector(
    removed_sequence: tuple = REMOVED_SEQUENCE, how_many_replacements: int = 6
):
    use_induced_relu = [
        [False] * 2 * 2
        for layer in [
            2,
        ]
        * 4
    ]
    for r in removed_sequence[:how_many_replacements]:
        row, col = layer_name_to_row_col(r)
        use_induced_relu[row][col] = True
    return use_induced_relu


def parse_args():
    parser = argparse.ArgumentParser(description="mmcls test model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "how_many_replacements",
        help="how many relu layers " "are replaced by " "Induced ReLUs",
        type=int,
    )
    parser.add_argument(
        "cache_root_dir",
        help="path to the cache root directory",
        type=str,
        default="osi_checkpoints/cifar100/resnet18",
    )

    parser.add_argument(
        "--removed_sequence",
        help="path to the removed sequence pickle",
        type=str,
        default=None,
    )

    parser.add_argument("--relu-spec-file", help="the relu spec file", default=None)
    parser.add_argument("--out", help="output result file")
    out_options = ["class_scores", "pred_score", "pred_label", "pred_class"]
    parser.add_argument(
        "--out-items",
        nargs="+",
        default=["all"],
        choices=out_options + ["none", "all"],
        help="Besides metrics, what items will be included in the output "
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        "above. Defaults to output all.",
        metavar="",
    )

    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where painted images will be saved"
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results",
    )
    parser.add_argument("--tmpdir", help="tmp dir for writing some results")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--metric-options",
        nargs="+",
        action=DictAction,
        default={},
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be parsed as a dict metric_options for dataset.evaluate()"
        " function.",
    )
    parser.add_argument(
        "--show-options",
        nargs="+",
        action=DictAction,
        help="custom options for show_result. key-value pair in xxx=yyy."
        "Check available options in `model.show_result`.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="(Deprecated, please use --gpu-id) ids of gpus to use "
        "(only applicable to non-distributed testing)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use " "(only applicable to non-distributed testing)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--device", help="device used for testing")
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def load_resnet18_choosing_matrices():
    with open(
        osp.join(
            "cached_relus/cifar100_with_val/resnet18/"
            "layer_name_to_choosing_matrix/"
            "layer_name_to_matrix.pkl"
        ),
        "rb",
    ) as f:
        layer_name_to_matrix = pickle.load(f)
    return layer_name_to_matrix


def load_permutation_matrices_for_resnet18():
    layer_name_to_matrix = load_resnet18_choosing_matrices()
    permutation_matrices = []
    for layer_index in [1, 2, 3, 4]:
        for sub_index in [0, 1]:
            for act_index in [1, 2]:
                layer_name = f"layer{layer_index}[{sub_index}].relu_{act_index}"
                permutation_matrices.append(layer_name_to_matrix[layer_name].cuda())
    return permutation_matrices


def get_model_cfg_based_on_bit_vector(bit_vector, permutation_matrices=None):
    if permutation_matrices is None:
        permutation_matrices = load_permutation_matrices_for_resnet18()
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
            permutation_matrices=permutation_matrices,
            use_induced_relu=bit_vector,
        ),
        neck=dict(type="GlobalAveragePooling"),
        head=dict(
            type="LinearClsHead",
            num_classes=100,
            in_channels=512,
            loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        ),
    )
    return model


def main(args, model_cfg=None, target_path: str = None):
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.relu_spec_file is not None:
        cfg.relu_spec_file = args.relu_spec_file
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(
            "`--gpu-ids` is deprecated, please use `--gpu-id`. "
            "Because we only support single GPU mode in "
            "non-distributed testing. Use the first GPU "
            "in `gpu_ids` now."
        )
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = args.device or auto_select_device()

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = False
        # distributed = True
        # try:
        #     init_dist(args.launcher, **cfg.dist_params)
        # except:
        #     pass
    if "distortion_extraction" in cfg.data:
        del cfg.data["distortion_extraction"]
    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if cfg.device == "ipu" else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update(
        {
            k: v
            for k, v in cfg.data.items()
            if k
            not in [
                "train",
                "val",
                "test",
                "train_dataloader",
                "val_dataloader",
                "test_dataloader",
            ]
        }
    )
    test_loader_cfg = {
        **loader_cfg,
        "shuffle": False,  # Not shuffle by default
        "sampler_cfg": None,  # Not use sampler by default
        **cfg.data.get("test_dataloader", {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect

    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    if model_cfg is None:
        model_cfg = cfg.model
    model = build_classifier(model_cfg)

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = torch.load(args.checkpoint)
    state = checkpoint["state_dict"]
    new_state = {}
    # duplicate parameters from head to backbone
    for k, v in state.items():
        new_state[k] = v
        if not k.startswith("backbone"):
            new_state["backbone" + k[len("head") :]] = v
    model.load_state_dict(new_state)
    checkpoint["state_dict"] = new_state
    checkpoint["meta"] = {"epoch": 180, "iter": 140760}
    torch.save(checkpoint, target_path)
    return model


if __name__ == "__main__":
    args = parse_args()
    if args.removed_sequence is None:
        removed_sequence = REMOVED_SEQUENCE
    else:
        with open(args.removed_sequence, "rb") as f:
            removed_sequence = pickle.load(f)

    bit_vector = create_use_induced_relu_bit_vector(
        removed_sequence, args.how_many_replacements
    )
    permutation_matrices = load_permutation_matrices_for_resnet18()
    model_cfg = get_model_cfg_based_on_bit_vector(bit_vector, permutation_matrices)
    target_path = os.path.join(
        args.cache_root_dir,
        f"replaced_{args.how_many_replacements}_relus_greedy",
        f"greedy_search_for_{args.how_many_replacements}_induced_relus.pth",
    )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    model = main(args, model_cfg=model_cfg, target_path=target_path)
