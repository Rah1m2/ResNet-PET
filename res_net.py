# author: detoX

from torch import nn
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """ 最小的残差块 """

    def __init__(self, input_channels, num_channels, is_use_1x1conv=False, stride=(1, 1), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2d(input_channels, num_channels, (3, 3), padding=1, stride=stride)
        self.conv2 = Conv2d(num_channels, num_channels, (3, 3), padding=1)
        self.bn1 = BatchNorm2d(num_channels)
        self.bn2 = BatchNorm2d(num_channels)
        if is_use_1x1conv:
            self.conv3 = Conv2d(input_channels, num_channels, (1, 1), padding=0, stride=(2, 2))
        else:
            self.conv3 = None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x  # 将上一步的结果直接与这一步的结果加起来，这样确保训练不会跑偏
        return F.relu(y)


def resnet_stage(input_channels, num_channels, num_block, is_first_stage=False):
    """ stage是一个大块，其中包含两个小残差块（BasicBlock） """
    blk = []
    for i in range(num_block):
        if i == 0 and not is_first_stage:  # 按照结构，一个Stage中有两个小残差块，如果运行的不是第一个stage并且是第一个小残差块的话，就需要用1x1卷积层升维
            blk.append(
                BasicBlock(input_channels, num_channels, is_use_1x1conv=True, stride=(2, 2)))
        else:
            blk.append(
                BasicBlock(num_channels, num_channels))
    return blk


class ResNet(nn.Module):
    """ 主网络 """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        b0 = Sequential(nn.Conv2d(47, 64, (7, 7), (2, 2), padding=3),
                        BatchNorm2d(64),
                        ReLU(),
                        MaxPool2d((3, 3), 2, 1))

        # resnet_stage函数返回值为元素为BasicBlock类实例的list，需要用*进行析构
        b1 = Sequential(*resnet_stage(64, 64, 2, is_first_stage=True))
        b2 = Sequential(*resnet_stage(64, 128, 2))
        b3 = Sequential(*resnet_stage(128, 256, 2))
        b4 = Sequential(*resnet_stage(256, 512, 2))
        self.sequential = Sequential(b0, b1, b2, b3, b4, AdaptiveAvgPool2d((1, 1)), Flatten(), Linear(512 * 1 * 1, 2))

    def forward(self, x):
        x = self.sequential(x)
        return x