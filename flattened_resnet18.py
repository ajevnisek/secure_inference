import torch
import torch.nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, Identity, Sequential


class FlattenedResNet18(torch.nn.Module):
    def __init__(self):
        super(FlattenedResNet18, self).__init__()
        self.conv0 = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn0 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)
        self.ResLayer0_BasicBlockV20_conv1 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer0_BasicBlockV20_norm1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer0_BasicBlockV20_relu_1 = ReLU(inplace=True)
        self.ResLayer0_BasicBlockV20_conv2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer0_BasicBlockV20_norm2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer0_BasicBlockV20_relu_2 = ReLU(inplace=True)
        self.ResLayer0_BasicBlockV20_downsample = None
        self.ResLayer0_BasicBlockV20_drop_path = Identity()
        self.ResLayer0_BasicBlockV21_conv1 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer0_BasicBlockV21_norm1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer0_BasicBlockV21_relu_1 = ReLU(inplace=True)
        self.ResLayer0_BasicBlockV21_conv2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer0_BasicBlockV21_norm2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer0_BasicBlockV21_relu_2 = ReLU(inplace=True)
        self.ResLayer0_BasicBlockV21_downsample = None
        self.ResLayer0_BasicBlockV21_drop_path = Identity()
        self.ResLayer1_BasicBlockV22_conv1 = Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.ResLayer1_BasicBlockV22_norm1 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer1_BasicBlockV22_relu_1 = ReLU(inplace=True)
        self.ResLayer1_BasicBlockV22_conv2 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer1_BasicBlockV22_norm2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer1_BasicBlockV22_relu_2 = ReLU(inplace=True)
        self.ResLayer1_BasicBlockV22_downsample = Sequential(*[Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False), BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)])
        self.ResLayer1_BasicBlockV22_drop_path = Identity()
        self.ResLayer1_BasicBlockV23_conv1 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer1_BasicBlockV23_norm1 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer1_BasicBlockV23_relu_1 = ReLU(inplace=True)
        self.ResLayer1_BasicBlockV23_conv2 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer1_BasicBlockV23_norm2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer1_BasicBlockV23_relu_2 = ReLU(inplace=True)
        self.ResLayer1_BasicBlockV23_downsample = None
        self.ResLayer1_BasicBlockV23_drop_path = Identity()
        self.ResLayer2_BasicBlockV24_conv1 = Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.ResLayer2_BasicBlockV24_norm1 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer2_BasicBlockV24_relu_1 = ReLU(inplace=True)
        self.ResLayer2_BasicBlockV24_conv2 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer2_BasicBlockV24_norm2 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer2_BasicBlockV24_relu_2 = ReLU(inplace=True)
        self.ResLayer2_BasicBlockV24_downsample = Sequential(*[Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False), BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)])
        self.ResLayer2_BasicBlockV24_drop_path = Identity()
        self.ResLayer2_BasicBlockV25_conv1 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer2_BasicBlockV25_norm1 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer2_BasicBlockV25_relu_1 = ReLU(inplace=True)
        self.ResLayer2_BasicBlockV25_conv2 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer2_BasicBlockV25_norm2 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer2_BasicBlockV25_relu_2 = ReLU(inplace=True)
        self.ResLayer2_BasicBlockV25_downsample = None
        self.ResLayer2_BasicBlockV25_drop_path = Identity()
        self.ResLayer3_BasicBlockV26_conv1 = Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.ResLayer3_BasicBlockV26_norm1 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer3_BasicBlockV26_relu_1 = ReLU(inplace=True)
        self.ResLayer3_BasicBlockV26_conv2 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer3_BasicBlockV26_norm2 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer3_BasicBlockV26_relu_2 = ReLU(inplace=True)
        self.ResLayer3_BasicBlockV26_downsample = Sequential(*[Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False), BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)])
        self.ResLayer3_BasicBlockV26_drop_path = Identity()
        self.ResLayer3_BasicBlockV27_conv1 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer3_BasicBlockV27_norm1 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer3_BasicBlockV27_relu_1 = ReLU(inplace=True)
        self.ResLayer3_BasicBlockV27_conv2 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.ResLayer3_BasicBlockV27_norm2 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ResLayer3_BasicBlockV27_relu_2 = ReLU(inplace=True)
        self.ResLayer3_BasicBlockV27_downsample = None
        self.ResLayer3_BasicBlockV27_drop_path = Identity()


    def forward(self, x):
        out = x
        out = self.conv0(out)
        out = self.bn0(out)
        out = self.relu0(out)
        ResLayer0_BasicBlockV20_identity = out
        out = self.ResLayer0_BasicBlockV20_conv1(out)
        out = self.ResLayer0_BasicBlockV20_norm1(out)
        out = self.ResLayer0_BasicBlockV20_relu_1(out)
        out = self.ResLayer0_BasicBlockV20_conv2(out)
        out = self.ResLayer0_BasicBlockV20_norm2(out)
        if self.ResLayer0_BasicBlockV20_downsample is not None: ResLayer0_BasicBlockV20_identity = self.ResLayer0_BasicBlockV20_downsample(ResLayer0_BasicBlockV20_identity)
        out = self.ResLayer0_BasicBlockV20_drop_path(out)
        out += ResLayer0_BasicBlockV20_identity
        out = self.ResLayer0_BasicBlockV20_relu_2(out)
        ResLayer0_BasicBlockV21_identity = out
        out = self.ResLayer0_BasicBlockV21_conv1(out)
        out = self.ResLayer0_BasicBlockV21_norm1(out)
        out = self.ResLayer0_BasicBlockV21_relu_1(out)
        out = self.ResLayer0_BasicBlockV21_conv2(out)
        out = self.ResLayer0_BasicBlockV21_norm2(out)
        if self.ResLayer0_BasicBlockV21_downsample is not None: ResLayer0_BasicBlockV21_identity = self.ResLayer0_BasicBlockV21_downsample(ResLayer0_BasicBlockV21_identity)
        out = self.ResLayer0_BasicBlockV21_drop_path(out)
        out += ResLayer0_BasicBlockV21_identity
        out = self.ResLayer0_BasicBlockV21_relu_2(out)
        ResLayer1_BasicBlockV22_identity = out
        out = self.ResLayer1_BasicBlockV22_conv1(out)
        out = self.ResLayer1_BasicBlockV22_norm1(out)
        out = self.ResLayer1_BasicBlockV22_relu_1(out)
        out = self.ResLayer1_BasicBlockV22_conv2(out)
        out = self.ResLayer1_BasicBlockV22_norm2(out)
        if self.ResLayer1_BasicBlockV22_downsample is not None: ResLayer1_BasicBlockV22_identity = self.ResLayer1_BasicBlockV22_downsample(ResLayer1_BasicBlockV22_identity)
        out = self.ResLayer1_BasicBlockV22_drop_path(out)
        out += ResLayer1_BasicBlockV22_identity
        out = self.ResLayer1_BasicBlockV22_relu_2(out)
        ResLayer1_BasicBlockV23_identity = out
        out = self.ResLayer1_BasicBlockV23_conv1(out)
        out = self.ResLayer1_BasicBlockV23_norm1(out)
        out = self.ResLayer1_BasicBlockV23_relu_1(out)
        out = self.ResLayer1_BasicBlockV23_conv2(out)
        out = self.ResLayer1_BasicBlockV23_norm2(out)
        if self.ResLayer1_BasicBlockV23_downsample is not None: ResLayer1_BasicBlockV23_identity = self.ResLayer1_BasicBlockV23_downsample(ResLayer1_BasicBlockV23_identity)
        out = self.ResLayer1_BasicBlockV23_drop_path(out)
        out += ResLayer1_BasicBlockV23_identity
        out = self.ResLayer1_BasicBlockV23_relu_2(out)
        ResLayer2_BasicBlockV24_identity = out
        out = self.ResLayer2_BasicBlockV24_conv1(out)
        out = self.ResLayer2_BasicBlockV24_norm1(out)
        out = self.ResLayer2_BasicBlockV24_relu_1(out)
        out = self.ResLayer2_BasicBlockV24_conv2(out)
        out = self.ResLayer2_BasicBlockV24_norm2(out)
        if self.ResLayer2_BasicBlockV24_downsample is not None: ResLayer2_BasicBlockV24_identity = self.ResLayer2_BasicBlockV24_downsample(ResLayer2_BasicBlockV24_identity)
        out = self.ResLayer2_BasicBlockV24_drop_path(out)
        out += ResLayer2_BasicBlockV24_identity
        out = self.ResLayer2_BasicBlockV24_relu_2(out)
        ResLayer2_BasicBlockV25_identity = out
        out = self.ResLayer2_BasicBlockV25_conv1(out)
        out = self.ResLayer2_BasicBlockV25_norm1(out)
        out = self.ResLayer2_BasicBlockV25_relu_1(out)
        out = self.ResLayer2_BasicBlockV25_conv2(out)
        out = self.ResLayer2_BasicBlockV25_norm2(out)
        if self.ResLayer2_BasicBlockV25_downsample is not None: ResLayer2_BasicBlockV25_identity = self.ResLayer2_BasicBlockV25_downsample(ResLayer2_BasicBlockV25_identity)
        out = self.ResLayer2_BasicBlockV25_drop_path(out)
        out += ResLayer2_BasicBlockV25_identity
        out = self.ResLayer2_BasicBlockV25_relu_2(out)
        ResLayer3_BasicBlockV26_identity = out
        out = self.ResLayer3_BasicBlockV26_conv1(out)
        out = self.ResLayer3_BasicBlockV26_norm1(out)
        out = self.ResLayer3_BasicBlockV26_relu_1(out)
        out = self.ResLayer3_BasicBlockV26_conv2(out)
        out = self.ResLayer3_BasicBlockV26_norm2(out)
        if self.ResLayer3_BasicBlockV26_downsample is not None: ResLayer3_BasicBlockV26_identity = self.ResLayer3_BasicBlockV26_downsample(ResLayer3_BasicBlockV26_identity)
        out = self.ResLayer3_BasicBlockV26_drop_path(out)
        out += ResLayer3_BasicBlockV26_identity
        out = self.ResLayer3_BasicBlockV26_relu_2(out)
        ResLayer3_BasicBlockV27_identity = out
        out = self.ResLayer3_BasicBlockV27_conv1(out)
        out = self.ResLayer3_BasicBlockV27_norm1(out)
        out = self.ResLayer3_BasicBlockV27_relu_1(out)
        out = self.ResLayer3_BasicBlockV27_conv2(out)
        out = self.ResLayer3_BasicBlockV27_norm2(out)
        if self.ResLayer3_BasicBlockV27_downsample is not None: ResLayer3_BasicBlockV27_identity = self.ResLayer3_BasicBlockV27_downsample(ResLayer3_BasicBlockV27_identity)
        out = self.ResLayer3_BasicBlockV27_drop_path(out)
        out += ResLayer3_BasicBlockV27_identity
        out = self.ResLayer3_BasicBlockV27_relu_2(out)
        return out
