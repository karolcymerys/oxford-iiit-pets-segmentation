from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn
from torch.nn.functional import interpolate
from torchvision import models
from torchvision.models import ResNet50_Weights


class Interpolate(nn.Module):
    def __init__(self, target_size: Tuple[int, int]) -> None:
        super(Interpolate, self).__init__()
        self.target_size = target_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return interpolate(features, size=self.target_size, mode='bilinear', align_corners=False)


class DeeplabV3(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super(DeeplabV3, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(OrderedDict([
            ('conv1', resnet.conv1),
            ('bn1', resnet.bn1),
            ('relu', nn.ReLU(inplace=True)),
            ('maxpool', resnet.maxpool),
            ('layer1', resnet.layer1),
            ('layer2', resnet.layer2),
            ('layer3', self.__layer3(resnet.layer3)),
            ('layer4', self.__layer4(resnet.layer4))
        ]))

        self.aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12),
                          bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(24, 24), dilation=(24, 24),
                          bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(36, 36), dilation=(36, 36),
                          bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Conv2d(2048, 256, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                Interpolate((28, 28))
            )
        ])
        self.project = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_labels, 1)
        )

    @staticmethod
    def __layer3(module: nn.Module):
        module[0].conv2.stride = (1, 1)
        module[0].downsample[0].stride = (1, 1)

        for idx, bottleneck_index in enumerate([1, 2, 3, 4, 5]):
            layer = module[bottleneck_index].conv2
            layer.stride = (1, 1)
            layer.padding = (2, 2)
            layer.dilation = (2, 2)
        return module

    @staticmethod
    def __layer4(module: nn.Module):
        module[0].conv2.stride = (1, 1)
        module[0].conv2.padding = (2, 2)
        module[0].conv2.dilation = (2, 2)
        module[0].downsample[0].stride = (1, 1)

        for idx, bottleneck_index in enumerate([1, 2]):
            layer = module[bottleneck_index].conv2
            layer.padding = (4, 4)
            layer.dilation = (4, 4)
        return module

    def init_weights(self) -> None:
        self.__init_weights(self.aspp)
        self.__init_weights(self.classifier)
        self.__init_weights(self.project)

    def __init_weights(self, module: nn.Module) -> None:
        for layer in module:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        backbone_output = self.backbone(features)
        classifier_output = torch.cat([_aspp(backbone_output) for _aspp in self.aspp], dim=1)
        projection_output = self.project(classifier_output)
        classifier_output = self.classifier(projection_output)
        return interpolate(classifier_output, features.size()[-2:], mode='bilinear', align_corners=False)
