from collections import OrderedDict
from functools import partial

import torch.nn as nn


class ConvBlock(nn.Module):
    """2D convolution followed by
         - an optional normalisation (batch norm or instance norm)
         - an optional activation (ReLU, LeakyReLU, or tanh)
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, norm='bn', activation='relu',
                 bias=False, transpose=False):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d, output_padding=1)
        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Invalid norm {}'.format(norm))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Invalid activation {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    """Residual block:
       x -> Conv -> norm -> act. -> Conv -> norm -> act. -> ADD -> out
         |                                                   |
          ---------------------------------------------------
    """
    def __init__(self, in_channels, out_channels=None, norm='bn', activation='relu', bias=False):
        super().__init__()
        out_channels = out_channels or in_channels

        self.layers = nn.Sequential(OrderedDict([
            ('conv_1', ConvBlock(in_channels, in_channels, 3, stride=1, norm=norm, activation=activation, bias=bias)),
            ('conv_2', ConvBlock(in_channels, out_channels, 3, stride=1, norm=norm, activation=activation, bias=bias)),
            ('dropout', nn.Dropout2d(0.25)),
        ]))

        if out_channels != in_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.projection = None

    def forward(self, x):
        x_residual = self.layers(x)

        if self.projection:
            x = self.projection(x)
        return x + x_residual
