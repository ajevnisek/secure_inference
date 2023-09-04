import torch

from mmcv import Config
from mmcls.models import build_classifier
from mmcls.datasets import build_dataloader

from mmcls.models.backbones.resnet import ResLayer
from mmcls.models.backbones.resnet_cifar_v2 import BasicBlockV2, BottleneckV2


def build_class(name, init_queue, forward_pass_queue):
    init_string = '\n'.join([f"        {s}" for s in init_queue])
    forward_string = '\n'.join([f'        {s}' for s in forward_pass_queue])
    class_str = f"""import torch
import torch.nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, Identity, Sequential


class {name}(torch.nn.Module):
    def __init__(self):
        super({name}, self).__init__()
{init_string}


    def forward(self, x):
{forward_string}
"""
    return class_str


def handle_downsample(downsample):
    if downsample is None:
        return None
    else:
        return f"Sequential(*{str(list(downsample))})"

def flatten_model(model):
    queue = []
    res_layer = 0
    basic_block = 0
    conv = 0
    bn = 0
    relu = 0
    forward_pass = ["out = x"]
    old_names_to_new_names = {}
    for child_name, child in model.backbone.named_children():
        if type(child) == torch.nn.Conv2d:
            queue.append(f"self.conv{conv} = {child}")
            forward_pass.append(f"out = self.conv{conv}(out)")
            old_names_to_new_names[child_name] =  f'conv{conv}'
            conv += 1
        elif type(child) == torch.nn.BatchNorm2d:
            queue.append(f"self.bn{bn} = {child}")
            forward_pass.append(f"out = self.bn{bn}(out)")
            old_names_to_new_names[child_name] =  f'bn{bn}'
            bn += 1
        elif type(child) == torch.nn.ReLU:
            queue.append(f"self.relu{relu} = {child}")
            forward_pass.append(f"out = self.relu{relu}(out)")
            relu += 1
        if type(child) == ResLayer:
            for sub_child_idx, sub_child in enumerate(list(child)):
                if type(sub_child) == BasicBlockV2:
                    prefix = f'ResLayer{res_layer}_BasicBlockV2{basic_block}'
                    old_names_to_new_names[f'{child_name}.{sub_child_idx}.conv1'] = f"{prefix}_conv1"
                    old_names_to_new_names[f'{child_name}.{sub_child_idx}.bn1'] = f"{prefix}_norm1"
                    old_names_to_new_names[f'{child_name}.{sub_child_idx}.conv2'] = f"{prefix}_conv2"
                    old_names_to_new_names[f'{child_name}.{sub_child_idx}.bn2'] = f"{prefix}_norm2"

                    queue.append(f"self.{prefix}_conv1 = {sub_child.conv1}")
                    queue.append(f"self.{prefix}_norm1 = {sub_child.norm1}")
                    queue.append(f"self.{prefix}_relu_1 = {sub_child.relu_1}")

                    queue.append(f"self.{prefix}_conv2 = {sub_child.conv2}")
                    queue.append(f"self.{prefix}_norm2 = {sub_child.norm2}")
                    queue.append(f"self.{prefix}_relu_2 = {sub_child.relu_2}")

                    queue.append(f"self.{prefix}_downsample = {handle_downsample(sub_child.downsample)}")
                    if handle_downsample(sub_child.downsample):
                        for sub_sub_child_idx, _ in enumerate(sub_child.downsample):
                            old_names_to_new_names[f'{child_name}.{sub_child_idx}.downsample.{sub_sub_child_idx}'
                            ] = f"{prefix}_downsample.{sub_sub_child_idx}"
                    queue.append(f"self.{prefix}_drop_path = {sub_child.drop_path}")

                    forward_pass.append(f"{prefix}_identity = out")
                    forward_pass.append(f"out = self.{prefix}_conv1(out)")
                    forward_pass.append(f"out = self.{prefix}_norm1(out)")
                    forward_pass.append(f"out = self.{prefix}_relu_1(out)")

                    forward_pass.append(f"out = self.{prefix}_conv2(out)")
                    forward_pass.append(f"out = self.{prefix}_norm2(out)")

                    forward_pass.append(
                        f"if self.{prefix}_downsample is not None: {prefix}_identity = self.{prefix}_downsample({prefix}_identity)")
                    forward_pass.append(f"out = self.{prefix}_drop_path(out)")
                    forward_pass.append(f"out += {prefix}_identity")
                    forward_pass.append(f"out = self.{prefix}_relu_2(out)")
                    basic_block += 1
                if type(sub_child) == BottleneckV2:
                    prefix = f'ResLayer{res_layer}_BottleneckV2{basic_block}'
                    old_names_to_new_names[f'{child_name}.{sub_child_idx}.conv1'] = f"{prefix}_conv1"
                    old_names_to_new_names[f'{child_name}.{sub_child_idx}.bn1'] = f"{prefix}_norm1"
                    old_names_to_new_names[f'{child_name}.{sub_child_idx}.conv2'] = f"{prefix}_conv2"
                    old_names_to_new_names[f'{child_name}.{sub_child_idx}.bn2'] = f"{prefix}_norm2"
                    old_names_to_new_names[f'{child_name}.{sub_child_idx}.conv3'] = f"{prefix}_conv3"
                    old_names_to_new_names[f'{child_name}.{sub_child_idx}.bn3'] = f"{prefix}_norm3"

                    queue.append(f"self.{prefix}_conv1 = {sub_child.conv1}")
                    queue.append(f"self.{prefix}_norm1 = {sub_child.norm1}")
                    queue.append(f"self.{prefix}_relu_1 = {sub_child.relu_1}")

                    queue.append(f"self.{prefix}_conv2 = {sub_child.conv2}")
                    queue.append(f"self.{prefix}_norm2 = {sub_child.norm2}")
                    queue.append(f"self.{prefix}_relu_2 = {sub_child.relu_2}")

                    queue.append(f"self.{prefix}_conv3 = {sub_child.conv3}")
                    queue.append(f"self.{prefix}_norm3 = {sub_child.norm3}")
                    queue.append(f"self.{prefix}_relu_3 = {sub_child.relu_3}")

                    queue.append(f"self.{prefix}_downsample = {handle_downsample(sub_child.downsample)}")
                    if handle_downsample(sub_child.downsample):
                        for sub_sub_child_idx, _ in enumerate(sub_child.downsample):
                            old_names_to_new_names[f'{child_name}.{sub_child_idx}.downsample.{sub_sub_child_idx}'
                            ] = f"{prefix}_downsample.{sub_sub_child_idx}"
                    queue.append(f"self.{prefix}_drop_path = {sub_child.drop_path}")

                    forward_pass.append(f"{prefix}_identity = out")
                    forward_pass.append(f"out = self.{prefix}_conv1(out)")
                    forward_pass.append(f"out = self.{prefix}_norm1(out)")
                    forward_pass.append(f"out = self.{prefix}_relu_1(out)")

                    forward_pass.append(f"out = self.{prefix}_conv2(out)")
                    forward_pass.append(f"out = self.{prefix}_norm2(out)")
                    forward_pass.append(f"out = self.{prefix}_relu_2(out)")

                    forward_pass.append(f"out = self.{prefix}_conv3(out)")
                    forward_pass.append(f"out = self.{prefix}_norm3(out)")

                    forward_pass.append(
                        f"if self.{prefix}_downsample is not None: {prefix}_identity = self.{prefix}_downsample({prefix}_identity)")
                    forward_pass.append(f"out = self.{prefix}_drop_path(out)")
                    forward_pass.append(f"out += {prefix}_identity")
                    forward_pass.append(f"out = self.{prefix}_relu_3(out)")

                    basic_block += 1
            res_layer += 1
    forward_pass.append("return out")
    return queue, forward_pass, old_names_to_new_names



def main():
    model_cfg = dict(
        type='ImageClassifier',
        backbone=dict(
            type='ResNet_CIFAR_V2',
            depth=18,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=100,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
    model = build_classifier(model_cfg)
    print(f"Original model backbone shape: {model.backbone(torch.rand(16, 3, 32, 32))[0].shape}")
    qqueue, fforward_pass, old_names_to_new_names = flatten_model(model)
    flattened_resnet18_decleration = build_class('FlattenedResNet18', qqueue, fforward_pass)
    with open('flattened_resnet18.py', 'w') as f:
        f.write(flattened_resnet18_decleration)

    # create a new checkpoint for the flattened resnet18 model
    pretrained_resnet18_checkpoint = torch.load('trained_networks/classification/resnet18_cifar100_resplit/baseline/latest.pth')
    old_state_dict = pretrained_resnet18_checkpoint['state_dict']
    new_state_dict = {}
    for k in old_state_dict.keys():
        if k.startswith('backbone'):
            element_name = '.'.join(k[len('backbone.'):].split('.')[:-1])
            data_name = k[len('backbone.'):].split('.')[-1]
            new_state_dict['backbone.' + old_names_to_new_names[element_name] + f'.{data_name}'] = \
            pretrained_resnet18_checkpoint['state_dict'][k]
        else:
            new_state_dict[k] = pretrained_resnet18_checkpoint['state_dict'][k]
    new_checkpoint = pretrained_resnet18_checkpoint
    new_checkpoint['state_dict'] = new_state_dict
    torch.save(new_checkpoint, '/mnt/data/secure_inference_cache/trained_networks/classification/resnet18_cifar100_resplit_flattened_backbone/baseline/latest.pth')
    #import ipdb; ipdb.set_trace()
    # pretrained_resnet18_checkpoint = torch.load(
    #     '/home/uriel/research/secure_inference/trained_networks/16384_finetune_attempt/latest.pth')
    # old_state_dict = pretrained_resnet18_checkpoint['state_dict']
    # new_state_dict = {}
    # for k in old_state_dict.keys():
    #     if k.startswith('backbone'):
    #         element_name = '.'.join(k[len('backbone.'):].split('.')[:-1])
    #         data_name = k[len('backbone.'):].split('.')[-1]
    #         new_state_dict['backbone.' + old_names_to_new_names[element_name] + f'.{data_name}'] = \
    #             pretrained_resnet18_checkpoint['state_dict'][k]
    #     else:
    #         new_state_dict[k] = pretrained_resnet18_checkpoint['state_dict'][k]
    # new_checkpoint = pretrained_resnet18_checkpoint
    # new_checkpoint['state_dict'] = new_state_dict
    # torch.save(new_checkpoint, 'flattened_resnet18_checkpoint_10_induced_relus.pth')


def convert_old_format_checkpoint_to_new_format_checkpoint(path_to_old_checkpoint, path_to_new_checkpoint):
    old_names_to_new_names = {'conv1': 'conv0', 'bn1': 'bn0', 'layer1.0.conv1': 'ResLayer0_BasicBlockV20_conv1', 'layer1.0.bn1': 'ResLayer0_BasicBlockV20_norm1', 'layer1.0.conv2': 'ResLayer0_BasicBlockV20_conv2', 'layer1.0.bn2': 'ResLayer0_BasicBlockV20_norm2', 'layer1.1.conv1': 'ResLayer0_BasicBlockV21_conv1', 'layer1.1.bn1': 'ResLayer0_BasicBlockV21_norm1', 'layer1.1.conv2': 'ResLayer0_BasicBlockV21_conv2', 'layer1.1.bn2': 'ResLayer0_BasicBlockV21_norm2', 'layer2.0.conv1': 'ResLayer1_BasicBlockV22_conv1', 'layer2.0.bn1': 'ResLayer1_BasicBlockV22_norm1', 'layer2.0.conv2': 'ResLayer1_BasicBlockV22_conv2', 'layer2.0.bn2': 'ResLayer1_BasicBlockV22_norm2', 'layer2.0.downsample.0': 'ResLayer1_BasicBlockV22_downsample.0', 'layer2.0.downsample.1': 'ResLayer1_BasicBlockV22_downsample.1', 'layer2.1.conv1': 'ResLayer1_BasicBlockV23_conv1', 'layer2.1.bn1': 'ResLayer1_BasicBlockV23_norm1', 'layer2.1.conv2': 'ResLayer1_BasicBlockV23_conv2', 'layer2.1.bn2': 'ResLayer1_BasicBlockV23_norm2', 'layer3.0.conv1': 'ResLayer2_BasicBlockV24_conv1', 'layer3.0.bn1': 'ResLayer2_BasicBlockV24_norm1', 'layer3.0.conv2': 'ResLayer2_BasicBlockV24_conv2', 'layer3.0.bn2': 'ResLayer2_BasicBlockV24_norm2', 'layer3.0.downsample.0': 'ResLayer2_BasicBlockV24_downsample.0', 'layer3.0.downsample.1': 'ResLayer2_BasicBlockV24_downsample.1', 'layer3.1.conv1': 'ResLayer2_BasicBlockV25_conv1', 'layer3.1.bn1': 'ResLayer2_BasicBlockV25_norm1', 'layer3.1.conv2': 'ResLayer2_BasicBlockV25_conv2', 'layer3.1.bn2': 'ResLayer2_BasicBlockV25_norm2', 'layer4.0.conv1': 'ResLayer3_BasicBlockV26_conv1', 'layer4.0.bn1': 'ResLayer3_BasicBlockV26_norm1', 'layer4.0.conv2': 'ResLayer3_BasicBlockV26_conv2', 'layer4.0.bn2': 'ResLayer3_BasicBlockV26_norm2', 'layer4.0.downsample.0': 'ResLayer3_BasicBlockV26_downsample.0', 'layer4.0.downsample.1': 'ResLayer3_BasicBlockV26_downsample.1', 'layer4.1.conv1': 'ResLayer3_BasicBlockV27_conv1', 'layer4.1.bn1': 'ResLayer3_BasicBlockV27_norm1', 'layer4.1.conv2': 'ResLayer3_BasicBlockV27_conv2', 'layer4.1.bn2': 'ResLayer3_BasicBlockV27_norm2'}
    pretrained_resnet18_checkpoint = torch.load(path_to_old_checkpoint)
    old_state_dict = pretrained_resnet18_checkpoint['state_dict']
    new_state_dict = {}
    for k in old_state_dict.keys():
        if k.startswith('backbone'):
            element_name = '.'.join(k[len('backbone.'):].split('.')[:-1])
            data_name = k[len('backbone.'):].split('.')[-1]
            if element_name in old_names_to_new_names:
                new_state_dict['backbone.' + old_names_to_new_names[element_name] + f'.{data_name}'] = \
                    pretrained_resnet18_checkpoint['state_dict'][k]
        else:
            new_state_dict[k] = pretrained_resnet18_checkpoint['state_dict'][k]
    new_checkpoint = pretrained_resnet18_checkpoint
    new_checkpoint['state_dict'] = new_state_dict
    torch.save(new_checkpoint, path_to_new_checkpoint)

if __name__ == '__main__':
    main()