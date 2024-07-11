from typing import Tuple, List

import torch
from torch import nn
from torch.nn import Parameter
from torchvision import models
from torchvision.models import VGG16_Weights

# https://www.kaggle.com/code/robinreni/pytorch-segnet-deeplabv3-training


def build_encoder_sub_layer(input_channels: int,
                            output_channels: int,
                            kernel_size: Tuple[int, int],
                            padding: Tuple[int, int]) -> List[nn.Module]:
    return [
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
    ]


class SegNetEncoderBlock(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 sub_layers_num: int,
                 conv_kernel_size: Tuple[int, int] = (3, 3),
                 conv_padding: Tuple[int, int] = (1, 1),
                 pooling_kernel_size: Tuple[int, int] = (2, 2),
                 pooling_padding: Tuple[int, int] = (0, 0)) -> None:
        super(SegNetEncoderBlock, self).__init__()

        layers = build_encoder_sub_layer(input_channels, output_channels, conv_kernel_size, conv_padding)
        for _ in range(sub_layers_num - 1):
            layers.extend(
                build_encoder_sub_layer(output_channels, output_channels, conv_kernel_size, conv_padding))
        layers.append(nn.MaxPool2d(pooling_kernel_size, pooling_kernel_size, pooling_padding))
        self.layers = nn.Sequential(*layers)

    def init_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.layers(features)


class SegNetDecoderBlock(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 sub_layers_num: int,
                 conv_kernel_size: Tuple[int, int] = (3, 3),
                 conv_padding: Tuple[int, int] = (1, 1),
                 transpose_kernel_size: Tuple[int, int] = (2, 2),
                 transpose_padding: Tuple[int, int] = (0, 0)) -> None:
        super(SegNetDecoderBlock, self).__init__()

        mid_channels = input_channels // 2
        layers = [nn.ConvTranspose2d(input_channels, input_channels, transpose_kernel_size, transpose_kernel_size, transpose_padding)]
        layers.extend(build_encoder_sub_layer(input_channels, mid_channels, conv_kernel_size, conv_padding))
        for _ in range(sub_layers_num - 2):
            layers.extend(
                build_encoder_sub_layer(mid_channels, mid_channels, conv_kernel_size, conv_padding))
        layers.extend(build_encoder_sub_layer(mid_channels, output_channels, conv_kernel_size, conv_padding))

        self.layers = nn.Sequential(*layers)

    def init_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class SegNet(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super(SegNet, self).__init__()

        self.encoder_layer_1 = SegNetEncoderBlock(3, 64, 2)
        self.encoder_layer_2 = SegNetEncoderBlock(64, 128, 2)
        self.encoder_layer_3 = SegNetEncoderBlock(128, 256, 3)
        self.encoder_layer_4 = SegNetEncoderBlock(256, 512, 3)
        self.encoder_layer_5 = SegNetEncoderBlock(512, 512, 3)

        self.decoder_layer_1 = SegNetDecoderBlock(512, 512, 3)
        self.decoder_layer_2 = SegNetDecoderBlock(1024, 256, 3)
        self.decoder_layer_3 = SegNetDecoderBlock(512, 128, 3)
        self.decoder_layer_4 = SegNetDecoderBlock(256, 64, 2)
        self.decoder_layer_5 = SegNetDecoderBlock(128, num_labels, 2)

    def init_weights(self) -> None:
        with torch.no_grad():
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

            self.encoder_layer_1.init_weights()
            self.encoder_layer_1.layers[0].load_state_dict(vgg.features[0].state_dict())
            self.encoder_layer_1.layers[3].load_state_dict(vgg.features[2].state_dict())

            self.encoder_layer_2.init_weights()
            self.encoder_layer_2.layers[0].load_state_dict(vgg.features[5].state_dict())
            self.encoder_layer_2.layers[3].load_state_dict(vgg.features[7].state_dict())

            self.encoder_layer_3.init_weights()
            self.encoder_layer_3.layers[0].load_state_dict(vgg.features[10].state_dict())
            self.encoder_layer_3.layers[3].load_state_dict(vgg.features[12].state_dict())
            self.encoder_layer_3.layers[6].load_state_dict(vgg.features[14].state_dict())

            self.encoder_layer_4.init_weights()
            self.encoder_layer_4.layers[0].load_state_dict(vgg.features[17].state_dict())
            self.encoder_layer_4.layers[3].load_state_dict(vgg.features[19].state_dict())
            self.encoder_layer_4.layers[6].load_state_dict(vgg.features[21].state_dict())

            self.encoder_layer_5.init_weights()
            self.encoder_layer_5.layers[0].load_state_dict(vgg.features[24].state_dict())
            self.encoder_layer_5.layers[3].load_state_dict(vgg.features[26].state_dict())
            self.encoder_layer_5.layers[6].load_state_dict(vgg.features[28].state_dict())

            self.decoder_layer_1.init_weights()
            self.decoder_layer_2.init_weights()
            self.decoder_layer_3.init_weights()
            self.decoder_layer_4.init_weights()
            self.decoder_layer_5.init_weights()

    def train_parameters(self) -> List[Parameter]:
        return [
            *self.decoder_layer_1.parameters(),
            *self.decoder_layer_2.parameters(),
            *self.decoder_layer_3.parameters(),
            *self.decoder_layer_4.parameters(),
            *self.decoder_layer_5.parameters()
        ]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        e_output_1 = self.encoder_layer_1(features)
        e_output_2 = self.encoder_layer_2(e_output_1)
        e_output_3 = self.encoder_layer_3(e_output_2)
        e_output_4 = self.encoder_layer_4(e_output_3)
        e_output_5 = self.encoder_layer_5(e_output_4)

        d_output_1 = self.decoder_layer_1(e_output_5)
        d_output_2 = self.decoder_layer_2(torch.cat([d_output_1, e_output_4], dim=1))
        d_output_3 = self.decoder_layer_3(torch.cat([d_output_2, e_output_3], dim=1))
        d_output_4 = self.decoder_layer_4(torch.cat([d_output_3, e_output_2], dim=1))
        return self.decoder_layer_5(torch.cat([d_output_4, e_output_1], dim=1))
