import torch
import torch.nn as nn
from typing import List
import torch.nn


class Generator(nn.Module):
    """_summary_

    Args:
        k_u - upsanpling ratet
        l - index of current upsampling block
    """

    def __init__(self, k_u, upsample_first, kernels, dilation):
        super(Generator, self).__init__()
        self.initial_ch = upsample_first
        self.conv1 = nn.Conv1d(
            80, self.initial_ch, kernel_size=7, dilation=1, padding=3
        )
        self.conv2 = nn.Conv1d(
            self.initial_ch // 2 ** (len(k_u)), 1, kernel_size=7, dilation=1, padding=3
        )
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.upsampling = nn.Sequential(
            *list(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        self.initial_ch // (2**i),
                        self.initial_ch // (2 ** (i + 1)),
                        kernel_size=k_u[i],
                        stride=k_u[i] // 2,
                        padding=(k_u[i] - (k_u[i] // 2)) // 2,
                    ),
                    MRF(
                        kernels,
                        dilation,
                        channels=self.initial_ch // (2 ** (i + 1)),
                    ),
                )
                for i in range(len(k_u))
            )
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.upsampling(x)
        x = self.tanh(self.conv2(x))
        return x


class MRF(nn.Module):
    """
    Composes ResBlocks with different kernels and dilations
    """

    def __init__(self, kernels: List, dilations: List, channels: int):
        super(MRF, self).__init__()
        self.resblocks = nn.ModuleList(
            list(
                ResBlock(
                    channels=channels,
                    kernel_size=kernels[i],
                    dilation=dilations[i],
                )
                for i in range(len(kernels))
            )
        )

    def forward(self, x):
        output = self.resblocks[0](x)
        for resblock in self.resblocks[1:]:
            output = output + resblock(x)
            # print(output.size())
        return output


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResBlock, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(len(dilation)):
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        dilation=dilation[i],
                        padding=int((kernel_size * dilation[i] - dilation[i]) / 2),
                    ),
                    nn.LeakyReLU(),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        dilation=1,
                        padding=int((kernel_size * 1 - 1) / 2),
                    ),
                )
            )

    def forward(self, x):
        for layer in self.convs:
            resid = x
            x = layer(x)
            x = x + resid
        return x
