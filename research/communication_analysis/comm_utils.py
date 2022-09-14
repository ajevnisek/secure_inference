# def run_model_block(self, model, activation, block_name):
#     if block_name == "stem":
#         activation = model.backbone.stem(activation)
#         activation = model.backbone.maxpool(activation)
#     elif block_name == "decode":
#         activation = model.decode_head([None, None, None, activation])
#     else:
#         res_layer_name, block_name = block_name.split("_")
#         layer = getattr(model.backbone, res_layer_name)
#         res_block = layer._modules[block_name]
#         activation = res_block(activation)
#
#     return activatio
import torch
import research
import pickle
from research.block_relu.utils import get_model, ArchUtilsFactory
from research.pipeline.backbones.secure_resnet import MyResNet



def get_conv_cost(tensor_shape, conv_module):
    # assert False, "Dilation?"
    assert conv_module.kernel_size[0] == conv_module.kernel_size[1]
    assert conv_module.stride[0] == conv_module.stride[1]
    assert tensor_shape[1] == conv_module.in_channels, f"{tensor_shape[1]}, {conv_module.in_channels}"

    stride = conv_module.stride[0]

    groups = conv_module.groups
    m = tensor_shape[0] / stride
    i = conv_module.in_channels
    o = conv_module.out_channels
    f = conv_module.kernel_size[0]
    q = tensor_shape[0] - f + 1
    if groups == 1:
        cost = 2 * m ** 2 * i + 2 * f ** 2 * o * i + q ** 2 * o
    else:
        assert groups == o == i
        cost = groups * (2 * m ** 2 + 2 * f ** 2 + q ** 2)
    return cost, [m, o]


def get_relu_cost(tensor_shape, relu_module):
    m = tensor_shape[0]
    log_p = 8

    if type(relu_module) == research.block_relu.utils.BlockRelu:
        assert tensor_shape[1] == len(relu_module.block_size_indices)
        communication_mult = 0
        communication_porthos_dReLU_bandwidth = 0

        for block_size_index in relu_module.block_size_indices:
            block_size = relu_module.block_sizes[block_size_index]
            bb = block_size[0] * block_size[1]
            communication_mult += (3 + 2 / bb) * m ** 2
            communication_porthos_dReLU_bandwidth += (6 * log_p + 19 - 5) * m ** 2 / bb

        return communication_mult + communication_porthos_dReLU_bandwidth, tensor_shape

    else:
        communication_cost = (5 * m ** 2 + (6 * log_p + 19 - 5) * m ** 2) * tensor_shape[1]
        return communication_cost, tensor_shape


def get_resnet_block_cost(block, tensor_shape):
    assert type(block) == research.pipeline.backbones.secure_resnet.SecureBottleneck
    tensor_shape_orig = [tensor_shape[0], tensor_shape[1]]
    cost_conv1, tensor_shape = get_conv_cost(tensor_shape, block.conv1)
    cost_relu1, tensor_shape = get_relu_cost(tensor_shape, block.relu_1)

    cost_conv2, tensor_shape = get_conv_cost(tensor_shape, block.conv2)
    cost_relu2, tensor_shape = get_relu_cost(tensor_shape, block.relu_2)

    cost_conv3, tensor_shape = get_conv_cost(tensor_shape, block.conv3)

    if block.downsample:
        cost_downsample, _ = get_conv_cost(tensor_shape_orig, block.downsample[0])
    else:
        cost_downsample = 0
    cost_relu3, tensor_shape = get_relu_cost(tensor_shape, block.relu_3)

    cost = (cost_conv1 + cost_relu1 + cost_conv2 + cost_relu2 + cost_conv3 + cost_relu3 + cost_downsample)
    return cost, tensor_shape


def get_resnet_stem_cost(model, tensor_shape):
    stem = model.backbone.stem
    cost_conv1, tensor_shape = get_conv_cost(tensor_shape, stem[0])
    cost_relu1, tensor_shape = get_relu_cost(tensor_shape, stem[2])
    cost_conv2, tensor_shape = get_conv_cost(tensor_shape, stem[3])
    cost_relu2, tensor_shape = get_relu_cost(tensor_shape, stem[5])
    cost_conv3, tensor_shape = get_conv_cost(tensor_shape, stem[6])
    cost_relu3, tensor_shape = get_relu_cost(tensor_shape, stem[8])

    cost = cost_conv1 + cost_relu1 + cost_conv2 + cost_relu2 + cost_conv3 + cost_relu3
    stride = model.backbone.maxpool.stride

    if type(model.backbone.maxpool) == torch.nn.modules.pooling.MaxPool2d:
        print('Hey')
        tensor_shape[0] /= stride
        log_p = 8
        cost += (model.backbone.maxpool.kernel_size - 1) * (6 * log_p + 24) * (tensor_shape[0] ** 2 * tensor_shape[1])
    else:
        tensor_shape[0] /= stride
    return cost, tensor_shape


def get_deeplab_decoder_cost(model, tensor_shape):
    cost = 0
    tensor_shape_orig = tensor_shape

    tensor_shapes = []
    cost_conv, tensor_shape = get_conv_cost(tensor_shape_orig, model.decode_head.image_pool[1].conv)
    cost_relu, _ = get_relu_cost(tensor_shape, model.decode_head.image_pool[1].activate)
    tensor_shapes.append(tensor_shape)
    cost += (cost_conv + cost_relu)
    for module in model.decode_head.aspp_modules:
        cost_conv, tensor_shape = get_conv_cost(tensor_shape_orig, module.conv)
        cost_relu, _ = get_relu_cost(tensor_shape, module.activate)
        cost += (cost_conv + cost_relu)
        tensor_shapes.append(tensor_shape)

    tensor_shape = (tensor_shapes[0][0], sum([x[1] for x in tensor_shapes]))

    cost_conv, tensor_shape = get_conv_cost(tensor_shape, model.decode_head.bottleneck.conv)
    cost_relu, tensor_shape = get_relu_cost(tensor_shape, model.decode_head.bottleneck.activate)
    cost_conv1, tensor_shape = get_conv_cost(tensor_shape, model.decode_head.conv_seg)
    cost += (cost_conv + cost_relu + cost_conv1)

    return cost


def get_resent_cost(model, tensor_shape):
    cost, tensor_shape = get_resnet_stem_cost(model, tensor_shape=tensor_shape)

    for layer_id in range(1, 5):
        layer = getattr(model.backbone, f"layer{layer_id}")
        for block in layer:
            cur_cost, tensor_shape = get_resnet_block_cost(block, tensor_shape=tensor_shape)
            cost += cur_cost


    cost += get_deeplab_decoder_cost(model, tensor_shape)
    return cost

def get_mobilenetv2_cost(model, tensor_shape):
    cost = 0

    cur_cost, tensor_shape = get_conv_cost(tensor_shape, model.backbone.conv1.conv)
    cost += cur_cost
    cur_cost, tensor_shape = get_relu_cost(tensor_shape, model.backbone.conv1.activate)
    cost += cur_cost

    for layer_id in range(1, 8):
        layer = getattr(model.backbone, f"layer{layer_id}")
        for block in layer:

            cur_cost, tensor_shape = get_conv_cost(tensor_shape, block.conv[0].conv)
            cost += cur_cost
            cur_cost, tensor_shape = get_relu_cost(tensor_shape, block.conv[0].activate)
            cost += cur_cost

            cur_cost, tensor_shape = get_conv_cost(tensor_shape, block.conv[1].conv)
            cost += cur_cost

            if layer_id > 1:
                cur_cost, tensor_shape = get_relu_cost(tensor_shape, block.conv[1].activate)
                cost += cur_cost

                cur_cost, tensor_shape = get_conv_cost(tensor_shape, block.conv[2].conv)
                cost += cur_cost
    cost += get_deeplab_decoder_cost(model, tensor_shape)
    return cost


# model = get_model(
#     config="/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/old/mobilenet_v2_ade_20k_baseline.py",
#     gpu_id=None,
#     checkpoint_path="/home/yakir/Data2/experiments/mobilenet_v2_ade_20k/baseline_relu6/latest.pth"
# )
#
# relu_spec_file = "/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/reduction_specs/layer_reduction_0.07.pickle"


# layer_name_to_block_size_indices = pickle.load(open(relu_spec_file, 'rb'))
# arch_utils = ArchUtilsFactory()("MobileNetV2")
# arch_utils.set_bReLU_layers(model, layer_name_to_block_size_indices)

# tensor_shape = [512, 3]
#
# print(get_mobilenetv2_cost(model, tensor_shape))
# 1138772825.7433875/6183918464.0
model = get_model(
    config="/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/resnet_coco_164K_2_hiers_a_pool_0.1.py",
    gpu_id=None,
    checkpoint_path="/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k/iter_80000.pth"
)
#
# relu_spec_file = "/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/V0/reduction_specs/layer_reduction_0.10.pickle"
#
#
# layer_name_to_block_size_indices = pickle.load(open(relu_spec_file, 'rb'))
# arch_utils = ArchUtilsFactory()("SecureResNet")
# arch_utils.set_bReLU_layers(model, layer_name_to_block_size_indices)
tensor_shape = [512, 3]
print(get_resent_cost(model, tensor_shape))



