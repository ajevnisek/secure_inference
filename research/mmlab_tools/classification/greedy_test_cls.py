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


DEFAULT_LAYER_NAME_TO_CHOOSING_MATRIX = (
    "cached_relus/cifar100_with_val/resnet18/layer_name_to_choosing_matrix/"
    "layer_name_to_matrix.pkl"
)


def parse_args():
    parser = argparse.ArgumentParser(description="mmcls test model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "layer_name_to_choosing_matrix_path",
        help="path to the pickle holding the mapping between "
        "layer names and the choosing matrix",
        default=DEFAULT_LAYER_NAME_TO_CHOOSING_MATRIX,
    )
    parser.add_argument(
        "results_dir", help="path to results directory", default="results/"
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
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="evaluation metrics, which depends on the dataset, e.g., "
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        "multi-label dataset",
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

    assert (
        args.metrics or args.out
    ), "Please specify at least one of output path and evaluation metrics."

    return args


def load_resnet18_choosing_matrices(
    layer_name_to_choosing_matrix_path=DEFAULT_LAYER_NAME_TO_CHOOSING_MATRIX,
):
    with open(osp.join(layer_name_to_choosing_matrix_path), "rb") as f:
        layer_name_to_matrix = pickle.load(f)
    return layer_name_to_matrix


def load_permutation_matrices_for_resnet18(layer_name_to_choosing_matrix_path):
    layer_name_to_matrix = load_resnet18_choosing_matrices(
        layer_name_to_choosing_matrix_path
    )
    permutation_matrices = []
    for layer_index in [1, 2, 3, 4]:
        for sub_index in [0, 1]:
            for act_index in [1, 2]:
                layer_name = f"layer{layer_index}[{sub_index}].relu_{act_index}"
                permutation_matrices.append(layer_name_to_matrix[layer_name].cuda())
    return permutation_matrices


def get_model_cfg_based_on_bit_vector(
    bit_vector,
    permutation_matrices=None,
    layer_name_to_choosing_matrix_path=DEFAULT_LAYER_NAME_TO_CHOOSING_MATRIX,
):
    if permutation_matrices is None:
        permutation_matrices = load_permutation_matrices_for_resnet18(
            layer_name_to_choosing_matrix_path
        )
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


def main(args, model_cfg=None):

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
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    if "CLASSES" in checkpoint.get("meta", {}):
        CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        from mmcls.datasets import ImageNet

        warnings.simplefilter("once")
        warnings.warn(
            "Class names are not saved in the checkpoint's "
            "meta data, use imagenet by default."
        )
        CLASSES = ImageNet.CLASSES

    if hasattr(cfg, "relu_spec_file") and cfg.relu_spec_file is not None:
        layer_name_to_block_sizes = pickle.load(open(cfg.relu_spec_file, "rb"))
        arch_utils = arch_utils_factory(cfg)
        arch_utils.set_bReLU_layers(model, layer_name_to_block_sizes)

    if not distributed:
        model = wrap_non_distributed_model(
            model, device=cfg.device, device_ids=cfg.gpu_ids
        )
        if cfg.device == "ipu":
            from mmcv.device.ipu import cfg2options, ipu_model_wrapper

            opts = cfg2options(cfg.runner.get("options_cfg", {}))
            if fp16_cfg is not None:
                model.half()
            model = ipu_model_wrapper(model, opts, fp16_cfg=fp16_cfg)
            data_loader.init(opts["inference"])
        model.CLASSES = CLASSES
        show_kwargs = args.show_options or {}
        outputs = single_gpu_test(
            model, data_loader, args.show, args.show_dir, **show_kwargs
        )
    else:
        model = wrap_distributed_model(
            model, device=cfg.device, broadcast_buffers=False
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    eval_results = None
    if rank == 0:
        results = {}
        logger = get_root_logger()
        if args.metrics:
            eval_results = dataset.evaluate(
                results=outputs,
                metric=args.metrics,
                metric_options=args.metric_options,
                logger=logger,
            )
            results.update(eval_results)
            for k, v in eval_results.items():
                if isinstance(v, np.ndarray):
                    v = [round(out, 2) for out in v.tolist()]
                elif isinstance(v, Number):
                    v = round(v, 2)
                else:
                    raise ValueError(f"Unsupport metric type: {type(v)}")
                print(f"\n{k} : {v}")
        if args.out:
            if "none" not in args.out_items:
                scores = np.vstack(outputs)
                pred_score = np.max(scores, axis=1)
                pred_label = np.argmax(scores, axis=1)
                pred_class = [CLASSES[lb] for lb in pred_label]
                res_items = {
                    "class_scores": scores,
                    "pred_score": pred_score,
                    "pred_label": pred_label,
                    "pred_class": pred_class,
                }
                if "all" in args.out_items:
                    results.update(res_items)
                else:
                    for key in args.out_items:
                        results[key] = res_items[key]
            print(f"\ndumping results to {args.out}")
            mmcv.dump(results, args.out)
    return eval_results


def tell_which_indices_were_not_converted_to_induced_relu(which_layer_to_turn_on):
    indices = []
    for i in range(which_layer_to_turn_on.shape[0]):
        for j in range(which_layer_to_turn_on.shape[1]):
            if not which_layer_to_turn_on[i, j]:
                indices.append(i * which_layer_to_turn_on.shape[0] + j)
    return indices


def greedy_search(get_accuracy_from_bit_vector):
    layer_names = [
        f"layer{layer_idx}[{layer_sub_index}].relu_{act_num}"
        for layer_idx in range(1, 4 + 1)
        for layer_sub_index in range(2)
        for act_num in range(1, 2 + 1)
    ]
    which_layer_to_turn_on = np.array([[False] * 4] * 4)
    layer_added = 0
    layer_added_to_layer_index_to_acc = {}
    indices = tell_which_indices_were_not_converted_to_induced_relu(
        which_layer_to_turn_on
    )
    print(indices)
    while len(indices) > 0:
        layer_name_to_acc = {}
        max_acc_index = -1
        max_acc = 0
        for layer_index in indices:
            print(f"Evaluating accuracy for: {layer_names[layer_index]}")
            which_layer_to_turn_on[np.unravel_index(layer_index, (4, 4))] = True
            acc = get_accuracy_from_bit_vector(which_layer_to_turn_on)
            if acc > max_acc:
                max_acc = acc
                max_acc_index = layer_index
            layer_name_to_acc[layer_names[layer_index]] = acc
            which_layer_to_turn_on[np.unravel_index(layer_index, (4, 4))] = False

        which_layer_to_turn_on[np.unravel_index(max_acc_index, (4, 4))] = True
        with open(
            os.path.join(
                args.results_dir,
                f"all_induced_relus_except_from_one_layer_name_to_acc_added_{layer_added + 1}"
                ".txt",
            ),
            "w",
        ) as f:
            json.dump(layer_name_to_acc, f, indent=4)
        layer_added_to_layer_index_to_acc[layer_added] = layer_name_to_acc
        layer_added += 1
        indices = tell_which_indices_were_not_converted_to_induced_relu(
            which_layer_to_turn_on
        )


def show_results_on_graph(original_model_accuracy=75.06, results_dir: str = "results"):
    files = [
        os.path.join(
            results_dir,
            f"all_induced_relus_except_from_one_layer_name_to_acc_added_{i}.txt",
        )
        for i in range(1, 16 + 1)
    ]

    acc = []
    layer_chosen = []
    for file in files:
        with open(file, "r") as f:
            d = json.load(f)
            acc.append(max(list(d.values())))
            layer_chosen.append(max(d.items(), key=lambda x: x[1]))

    import matplotlib.pyplot as plt

    plt.clf()
    plt.plot(range(1, 16 + 1), acc, "-x", linewidth=3)
    plt.plot(range(1, 16 + 1), [original_model_accuracy] * 16, "--k", linewidth=3)
    plt.xticks(range(1, 16 + 1), [x[0] for x in layer_chosen], rotation=90)
    plt.xlabel("layer replaced")
    plt.ylabel("accuracy [%]")
    plt.title("greedy search of ReLU to Induced ReLU swaps")
    plt.grid(True)
    plt.legend(["greedy", "baseline=all ReLUs"])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "greedy_search_of_induced_relus.png"))
    # plt.show()


if __name__ == "__main__":
    args = parse_args()
    original_model_accuracies = main(args)
    original_model_top_1_accuracy = original_model_accuracies["accuracy_top-1"]
    permutation_matrices = load_permutation_matrices_for_resnet18(
        args.layer_name_to_choosing_matrix_path
    )
    bit_vector = [
        [True] * 2 * 2
        for layer in [
            2,
        ]
        * 4
    ]
    bit_vector[0][0] = True

    def get_accuracy_from_bit_vector(bit_vector):
        model_cfg = get_model_cfg_based_on_bit_vector(
            bit_vector, permutation_matrices, args.layer_name_to_choosing_matrix_path
        )
        result = main(args, model_cfg)
        accuracy = result["accuracy_top-1"]
        return accuracy

    print(f"accuracy = {get_accuracy_from_bit_vector(bit_vector)}")
    greedy_search(get_accuracy_from_bit_vector)
    show_results_on_graph(original_model_top_1_accuracy, args.results_dir)
