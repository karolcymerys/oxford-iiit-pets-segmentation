from typing import Tuple

import torch
from torch import nn

from unet.utils import crop


class ContractingBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 padding: Tuple[int, int] = (0, 0),
                 apply_max_pooling: bool = True,
                 pooling_kernel_size: Tuple[int, int] = (2, 2)) -> None:
        super(ContractingBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),  # Paper doesn't mention it
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.apply_max_pooling = apply_max_pooling
        if apply_max_pooling:
            self.pooling = nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.block(x)

        return self.pooling(output) if self.apply_max_pooling else output, output

    def init_weights(self) -> None:
        for layer in self.block:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()


class ExpansiveBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 padding: Tuple[int, int] = (0, 0),
                 upconv_kernel_size: Tuple[int, int] = (2, 2)) -> None:
        super(ExpansiveBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=upconv_kernel_size,
                                         stride=upconv_kernel_size)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),  # Paper doesn't mention it
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, contracting_input: torch.Tensor) -> torch.Tensor:
        upconv_output = self.upconv(x)
        cropped_contracting_input = crop(contracting_input, upconv_output.shape)
        return self.block(torch.cat([upconv_output, cropped_contracting_input], dim=1))

    def init_weights(self) -> None:
        for layer in self.block:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()


class Unet(nn.Module):
    def __init__(self, num_labels: int = 2) -> None:
        super(Unet, self).__init__()

        self.contracting_block_1 = ContractingBlock(3, 64)
        self.contracting_block_2 = ContractingBlock(64, 128)
        self.contracting_block_3 = ContractingBlock(128, 256)
        self.contracting_block_4 = ContractingBlock(256, 512)
        self.contracting_block_5 = ContractingBlock(512, 1024, apply_max_pooling=False)

        self.expansive_block_1 = ExpansiveBlock(1024, 512)
        self.expansive_block_2 = ExpansiveBlock(512, 256)
        self.expansive_block_3 = ExpansiveBlock(256, 128)
        self.expansive_block_4 = ExpansiveBlock(128, 64)

        self.classifier = nn.Conv2d(64, num_labels, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cb1_output, cb1_output_copy = self.contracting_block_1(x)
        cb2_output, cb2_output_copy = self.contracting_block_2(cb1_output)
        cb3_output, cb3_output_copy = self.contracting_block_3(cb2_output)
        cb4_output, cb4_output_copy = self.contracting_block_4(cb3_output)
        cb5_output, _ = self.contracting_block_5(cb4_output)

        eb1_output = self.expansive_block_1(cb5_output, cb4_output_copy)
        eb2_output = self.expansive_block_2(eb1_output, cb3_output_copy)
        eb3_output = self.expansive_block_3(eb2_output, cb2_output_copy)
        eb4_output = self.expansive_block_4(eb3_output, cb1_output_copy)

        return self.classifier(eb4_output)

    def init_weights(self) -> None:
        self.contracting_block_1.init_weights()
        self.contracting_block_2.init_weights()
        self.contracting_block_3.init_weights()
        self.contracting_block_4.init_weights()
        self.contracting_block_5.init_weights()
        self.expansive_block_1.init_weights()
        self.expansive_block_2.init_weights()
        self.expansive_block_3.init_weights()
        self.expansive_block_4.init_weights()

        nn.init.kaiming_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
