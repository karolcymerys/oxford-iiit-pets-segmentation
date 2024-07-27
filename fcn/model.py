from typing import Tuple, List

import torch
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights


def fcn_layer(input_channels: int,
              output_channels: int,
              kernel_size: Tuple[int, int],
              padding: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(*fcn_layer_as_list(input_channels, output_channels, kernel_size, padding))


def fcn_layer_as_list(input_channels: int,
                      output_channels: int,
                      kernel_size: Tuple[int, int],
                      padding: Tuple[int, int]) -> List[nn.Module]:
    return [
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
    ]


def crop(features: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    features_size = features.size()
    h_start = (features_size[2] - size[0]) // 2
    w_start = (features_size[3] - size[1]) // 2
    return features[:, :, h_start:h_start+size[0], w_start:w_start+size[1]]


class FCNEncoderBlock(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 sub_layers_num: int,
                 conv_kernel_size: Tuple[int, int] = (3, 3),
                 first_conv_padding: Tuple[int, int] = (1, 1),
                 conv_padding: Tuple[int, int] = (1, 1),
                 pooling_kernel_size: Tuple[int, int] = (2, 2),
                 pooling_padding: Tuple[int, int] = (0, 0)) -> None:
        super(FCNEncoderBlock, self).__init__()

        layers = fcn_layer_as_list(input_channels, output_channels, conv_kernel_size, first_conv_padding)
        for _ in range(sub_layers_num - 1):
            layers.extend(
                fcn_layer_as_list(output_channels, output_channels, conv_kernel_size, conv_padding))

        layers.append(nn.MaxPool2d(pooling_kernel_size, pooling_kernel_size, pooling_padding))
        self.layers = nn.Sequential(*layers)

    def init_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class FCN32s(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super(FCN32s, self).__init__()
        self.layer_1 = FCNEncoderBlock(3, 64, 2, first_conv_padding=(100, 100))
        self.layer_2 = FCNEncoderBlock(64, 128, 2)
        self.layer_3 = FCNEncoderBlock(128, 256, 3)
        self.layer_4 = FCNEncoderBlock(256, 512, 3)
        self.layer_5 = FCNEncoderBlock(512, 512, 3)
        self.layer_6 = fcn_layer(512, 4096, (7, 7), (0, 0))
        self.layer_7 = fcn_layer(4096, 4096, (1, 1), (0, 0))

        self.classifier = nn.Conv2d(4096, num_labels, kernel_size=(1, 1), padding=(0, 0))
        self.up_sample = nn.ConvTranspose2d(num_labels, num_labels, kernel_size=(64, 64), stride=(32, 32), bias=False)

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        self.layer_1.init_weights()
        self.layer_1.layers[0].load_state_dict(vgg.features[0].state_dict())
        self.layer_1.layers[3].load_state_dict(vgg.features[2].state_dict())

        self.layer_2.init_weights()
        self.layer_2.layers[0].load_state_dict(vgg.features[5].state_dict())
        self.layer_2.layers[3].load_state_dict(vgg.features[7].state_dict())

        self.layer_3.init_weights()
        self.layer_3.layers[0].load_state_dict(vgg.features[10].state_dict())
        self.layer_3.layers[3].load_state_dict(vgg.features[12].state_dict())
        self.layer_3.layers[6].load_state_dict(vgg.features[14].state_dict())

        self.layer_4.init_weights()
        self.layer_4.layers[0].load_state_dict(vgg.features[17].state_dict())
        self.layer_4.layers[3].load_state_dict(vgg.features[19].state_dict())
        self.layer_4.layers[6].load_state_dict(vgg.features[21].state_dict())

        self.layer_5.init_weights()
        self.layer_5.layers[0].load_state_dict(vgg.features[24].state_dict())
        self.layer_5.layers[3].load_state_dict(vgg.features[26].state_dict())
        self.layer_5.layers[6].load_state_dict(vgg.features[28].state_dict())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        input_size = features.size()
        output_1 = self.layer_1(features)
        output_2 = self.layer_2(output_1)
        output_3 = self.layer_3(output_2)
        output_4 = self.layer_4(output_3)
        output_5 = self.layer_5(output_4)
        output_6 = self.layer_6(output_5)
        output_7 = self.layer_7(output_6)

        output_classifier = self.classifier(output_7)
        up_sampled = self.up_sample(output_classifier)
        return crop(up_sampled, (input_size[2], input_size[3]))


class FCN16s(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super(FCN16s, self).__init__()
        self.layer_1 = FCNEncoderBlock(3, 64, 2, first_conv_padding=(100, 100))
        self.layer_2 = FCNEncoderBlock(64, 128, 2)
        self.layer_3 = FCNEncoderBlock(128, 256, 3)
        self.layer_4 = FCNEncoderBlock(256, 512, 3)
        self.layer_5 = FCNEncoderBlock(512, 512, 3)
        self.layer_6 = fcn_layer(512, 4096, (7, 7), (0, 0))
        self.layer_7 = fcn_layer(4096, 4096, (1, 1), (0, 0))

        self.classifier_layer_7 = nn.Conv2d(4096, num_labels, kernel_size=(1, 1), padding=(0, 0))
        self.classifier_layer_4 = nn.Conv2d(512, num_labels, kernel_size=(1, 1), padding=(0, 0))

        self.up_sample_2_layer = nn.ConvTranspose2d(num_labels, num_labels, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.up_sample_16_layer = nn.ConvTranspose2d(num_labels, num_labels, kernel_size=(32, 32), stride=(16, 16), bias=False)

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        self.layer_1.init_weights()
        self.layer_1.layers[0].load_state_dict(vgg.features[0].state_dict())
        self.layer_1.layers[3].load_state_dict(vgg.features[2].state_dict())

        self.layer_2.init_weights()
        self.layer_2.layers[0].load_state_dict(vgg.features[5].state_dict())
        self.layer_2.layers[3].load_state_dict(vgg.features[7].state_dict())

        self.layer_3.init_weights()
        self.layer_3.layers[0].load_state_dict(vgg.features[10].state_dict())
        self.layer_3.layers[3].load_state_dict(vgg.features[12].state_dict())
        self.layer_3.layers[6].load_state_dict(vgg.features[14].state_dict())

        self.layer_4.init_weights()
        self.layer_4.layers[0].load_state_dict(vgg.features[17].state_dict())
        self.layer_4.layers[3].load_state_dict(vgg.features[19].state_dict())
        self.layer_4.layers[6].load_state_dict(vgg.features[21].state_dict())

        self.layer_5.init_weights()
        self.layer_5.layers[0].load_state_dict(vgg.features[24].state_dict())
        self.layer_5.layers[3].load_state_dict(vgg.features[26].state_dict())
        self.layer_5.layers[6].load_state_dict(vgg.features[28].state_dict())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        input_size = features.size()
        output_1 = self.layer_1(features)
        output_2 = self.layer_2(output_1)
        output_3 = self.layer_3(output_2)
        output_4 = self.layer_4(output_3)
        output_5 = self.layer_5(output_4)
        output_6 = self.layer_6(output_5)
        output_7 = self.layer_7(output_6)

        output_classifier_layer_7 = self.classifier_layer_7(output_7)
        up_sampled_layer_7 = self.up_sample_2_layer(output_classifier_layer_7)
        up_sampled_layer_7_size = up_sampled_layer_7.size()

        output_classifier_layer_4 = crop(self.classifier_layer_4(output_4),
                                         (up_sampled_layer_7_size[2], up_sampled_layer_7_size[3]))

        result = self.up_sample_16_layer(output_classifier_layer_4 + up_sampled_layer_7)
        return crop(result, (input_size[2], input_size[3]))


class FCN8s(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super(FCN8s, self).__init__()
        self.layer_1 = FCNEncoderBlock(3, 64, 2, first_conv_padding=(100, 100))
        self.layer_2 = FCNEncoderBlock(64, 128, 2)
        self.layer_3 = FCNEncoderBlock(128, 256, 3)
        self.layer_4 = FCNEncoderBlock(256, 512, 3)
        self.layer_5 = FCNEncoderBlock(512, 512, 3)
        self.layer_6 = fcn_layer(512, 4096, (7, 7), (0, 0))
        self.layer_7 = fcn_layer(4096, 4096, (1, 1), (0, 0))

        self.classifier_layer_7 = nn.Conv2d(4096, num_labels, kernel_size=(1, 1), padding=(0, 0))
        self.classifier_layer_4 = nn.Conv2d(512, num_labels, kernel_size=(1, 1), padding=(0, 0))
        self.classifier_layer_3 = nn.Conv2d(256, num_labels, kernel_size=(1, 1), padding=(0, 0))

        self.up_sample_2_layer_7 = nn.ConvTranspose2d(num_labels, num_labels, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.up_sample_2_layer_4 = nn.ConvTranspose2d(num_labels, num_labels, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.up_sample_16_layer_3 = nn.ConvTranspose2d(num_labels, num_labels, kernel_size=(32, 32), stride=(16, 16), bias=False)

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        self.layer_1.init_weights()
        self.layer_1.layers[0].load_state_dict(vgg.features[0].state_dict())
        self.layer_1.layers[3].load_state_dict(vgg.features[2].state_dict())

        self.layer_2.init_weights()
        self.layer_2.layers[0].load_state_dict(vgg.features[5].state_dict())
        self.layer_2.layers[3].load_state_dict(vgg.features[7].state_dict())

        self.layer_3.init_weights()
        self.layer_3.layers[0].load_state_dict(vgg.features[10].state_dict())
        self.layer_3.layers[3].load_state_dict(vgg.features[12].state_dict())
        self.layer_3.layers[6].load_state_dict(vgg.features[14].state_dict())

        self.layer_4.init_weights()
        self.layer_4.layers[0].load_state_dict(vgg.features[17].state_dict())
        self.layer_4.layers[3].load_state_dict(vgg.features[19].state_dict())
        self.layer_4.layers[6].load_state_dict(vgg.features[21].state_dict())

        self.layer_5.init_weights()
        self.layer_5.layers[0].load_state_dict(vgg.features[24].state_dict())
        self.layer_5.layers[3].load_state_dict(vgg.features[26].state_dict())
        self.layer_5.layers[6].load_state_dict(vgg.features[28].state_dict())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        input_size = features.size()
        output_1 = self.layer_1(features)
        output_2 = self.layer_2(output_1)
        output_3 = self.layer_3(output_2)
        output_4 = self.layer_4(output_3)
        output_5 = self.layer_5(output_4)
        output_6 = self.layer_6(output_5)
        output_7 = self.layer_7(output_6)

        output_classifier_layer_7 = self.classifier_layer_7(output_7)
        up_sampled_layer_7 = self.up_sample_2_layer_7(output_classifier_layer_7)
        up_sampled_layer_7_size = up_sampled_layer_7.size()

        output_classifier_layer_4 = crop(self.classifier_layer_4(output_4),
                                         (up_sampled_layer_7_size[2], up_sampled_layer_7_size[3]))

        up_sampled_layer_4_7 = self.up_sample_2_layer_4(output_classifier_layer_4 + up_sampled_layer_7)
        cropped_and_up_sampled_layer_4_7 = crop(up_sampled_layer_4_7, (input_size[2], input_size[3]))
        up_sampled_layer_4_size = cropped_and_up_sampled_layer_4_7.size()

        output_classifier_layer_3 = crop(self.classifier_layer_3(output_3),
                                         (up_sampled_layer_4_size[2], up_sampled_layer_4_size[3]))

        return crop(self.up_sample_16_layer_3(output_classifier_layer_3 + cropped_and_up_sampled_layer_4_7),
                    (input_size[2], input_size[3]))
