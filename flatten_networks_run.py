import torch
from flattened_resnet18 import FlattenedResNet18
from mmcls.models import build_classifier

net = FlattenedResNet18()
print(net(torch.rand(16, 3, 32, 32)).shape)

model_cfg = dict(
        type='ImageClassifier',
        backbone=dict(
            type='FlattenedResNet18',
        ),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=100,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
model = build_classifier(model_cfg)

