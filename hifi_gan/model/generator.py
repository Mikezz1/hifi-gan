import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class Generator(nn.Module):
    """_summary_

    Args:
        k_u - upsanpling ratet
        l - index of current upsampling block
    """

    def __init__(self, k_u, upsample_first, kernels, dilation):
        super(Generator, self).__init__()
        self.initial_ch = upsample_first
        self.norm = weight_norm
        self.conv1 = self.norm(
            nn.Conv1d(80, self.initial_ch, kernel_size=7, dilation=1, padding=3)
        )
        self.conv1.apply(init_weights)
        self.conv2 = self.norm(
            nn.Conv1d(
                self.initial_ch // 2 ** (len(k_u)),
                1,
                kernel_size=7,
                dilation=1,
                padding=3,
            )
        )
        self.conv2.apply(init_weights)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.upsampling = nn.Sequential(
            *list(
                nn.Sequential(
                    nn.LeakyReLU(),
                    self.norm(
                        nn.ConvTranspose1d(
                            self.initial_ch // (2**i),
                            self.initial_ch // (2 ** (i + 1)),
                            kernel_size=k_u[i],
                            stride=k_u[i] // 2,
                            padding=k_u[i] // 4,
                        ),
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
        self.upsampling.apply(init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.upsampling(x)
        x = F.leaky_relu(x)
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
        output = torch.zeros(x.size()).to(x.device)
        for resblock in self.resblocks:
            output = output + resblock(x)
        return output / len(self.resblocks)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResBlock, self).__init__()
        self.convs = nn.ModuleList()
        self.norm = weight_norm
        for i in range(len(dilation)):
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(),
                    self.norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size=kernel_size,
                            dilation=dilation[i],
                            padding=int((kernel_size * dilation[i] - dilation[i]) / 2),
                        )
                    ),
                    nn.LeakyReLU(),
                    self.norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size=kernel_size,
                            dilation=1,
                            padding=int((kernel_size * 1 - 1) / 2),
                        )
                    ),
                )
            )
        self.convs.apply(init_weights)

    def forward(self, x):
        for layer in self.convs:
            resid = x
            x = layer(x)
            x = x + resid
        return x
