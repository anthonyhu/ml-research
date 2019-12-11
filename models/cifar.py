from collections import OrderedDict
import torch.nn as nn
from torchvision.models import resnet18, vgg11_bn

from layers.convolutions import ConvBlock, ResBlock
from layers.preprocessing import ImagePreprocessing


class CifarModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = nn.Sequential(OrderedDict([
        #     ('image_preprocessing', ImagePreprocessing()),
        #     ('conv_1', ConvBlock(3, 8)),
        #     ('res_1_1', ResBlock(8, 16)),
        #     ('res_1_2', ResBlock(16, 16)),
        #     ('res_1_3', ResBlock(16, 32)),
        #     ('conv_2', ConvBlock(32, 64, stride=2)),
        #     ('res_2', ResBlock(64, 64)),
        #     ('conv_3', ConvBlock(64, 128, stride=2)),
        #     ('res_3', ResBlock(128, 128)),
        #     ('conv_4', ConvBlock(128, 256, stride=2)),
        #     ('res_4', ResBlock(256, 256)),
        #     ('avg_pool', nn.AdaptiveAvgPool2d(1)),
        #     ('flatten', nn.Flatten()),
        #     ('fc_1_linear', nn.Linear(256, 50)),
        #     ('fc_1_norm', nn.BatchNorm1d(50)),
        #     ('fc_1_act', nn.ReLU()),
        #     ('out', nn.Linear(50, 10)),
        # ]))

        self.model = nn.Sequential(OrderedDict([
            ('image_preprocessing', ImagePreprocessing()),
            ('conv_1_1', ConvBlock(3, 8)),
            ('conv_1_2', ConvBlock(8, 16)),
            ('conv_1_3', ConvBlock(16, 32)),
            ('conv_1_4', ConvBlock(32, 64)),
            ('conv_2', ConvBlock(64, 128, stride=2)),
            ('conv_3', ConvBlock(128, 256, stride=2)),
            ('conv_4', ConvBlock(256, 512, stride=2)),
            ('avg_pool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten()),
            ('fc_1_linear', nn.Linear(512, 100)),
            ('fc_1_act', nn.ReLU()),
            ('out', nn.Linear(100, 10)),
        ]))

    def forward(self, x):
        return self.model(x)
