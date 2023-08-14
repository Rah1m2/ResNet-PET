# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import torch
import glob
import torchvision.transforms
import SimpleITK as sitk
from torch import nn
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from torch.nn import functional as F
import nibabel as nib
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


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
        b0 = Sequential(nn.Conv2d(50, 64, (7, 7), (2, 2), padding=3),
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


DATA_CACHE = {}


class NiiDataset(Dataset):
    classes = [
        "0 - NC",
        "1 - MCI",
    ]

    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            img = nib.load(self.img_path[index])
            img = img.dataobj[:, :, :, 0]
            DATA_CACHE[self.img_path[index]] = img

        idx = np.random.choice(range(img.shape[-1]), 50)
        # idx.sort()
        img = img[:, :, idx]
        img = img.astype(np.float32)

        # img /= 255.0
        # img -= 1

        # print(img.shape)
        if self.transform is not None:
            img = self.transform(image=img)['image']  # 用A进行transform之后返回一个字典，有两个属性，其中一个image属性就是图片

        img = img.transpose([2, 0, 1])
        # 如果这个nii文件是正常人的，label就标记为True，然后再将bool转化为int，也就是1
        return img, torch.from_numpy(np.array(int('NC' in self.img_path[index])))

    def __len__(self):
        return len(self.img_path)


def test():
    test_list = [1, 2, 3, 4, 5]
    print(*test_list)

    # enumerate返回值结构展示
    for i, (ob1, ob2) in enumerate([['img', 'target'], ['civilian', 'combine']]):
        print("enum test:")
        print(ob1)
        print(ob2)
        print(i)

    # 每层网络shape
    X = torch.rand(size=(1, 1, 224, 224), dtype=torch.float32)
    rn_net = ResNet()
    for layer in rn_net.sequential:
        X = layer(X)
        print(layer.__class__.__name__, "output shape:\t", X.shape)


def main():
    # 选择使用gpu还是cpu进行训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------初始化tensorboard---------
    writer = SummaryWriter("logs")

    # ---------加载数据集---------
    # 数据集路径
    train_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Train\\*\\*")
    test_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Test\\*")
    print(train_path[:-10])
    print(train_path)
    # 随机打乱
    np.random.shuffle(train_path)
    np.random.shuffle(test_path)
    batch_size = 4
    # 训练集dataloader
    train_loader = torch.utils.data.DataLoader(
        NiiDataset(train_path,
                   A.Compose([
                       A.RandomRotate90(),
                       A.RandomCrop(120, 120),
                       A.HorizontalFlip(p=0.5),
                       A.RandomContrast(p=0.5),
                       A.RandomBrightnessContrast(p=0.5),
                   ])
                   ), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True
    )
    # 测试集dataloader
    test_loader = torch.utils.data.DataLoader(
        NiiDataset(test_path,
                   A.Compose([
                       A.RandomCrop(128, 128),
                       A.HorizontalFlip(p=0.5),
                       A.RandomContrast(p=0.5),
                   ])
                   ), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=True
    )
    # len(train_loader)得出的结果是batch的数量，必须乘以batch_size才是数据的总数
    train_set_len = len(train_loader) * batch_size
    test_set_len = len(test_loader) * batch_size

    # ---------初始化网络---------
    # 神经网络模型
    rn_module = ResNet()
    rn_module = rn_module.to(device)
    # 损失函数，使用GPU进行训练
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)
    # 优化器
    learning_rate = 0.01
    optimizer = torch.optim.SGD(rn_module.parameters(), lr=learning_rate)

    # ---------初始化训练相关参数---------
    # 训练次数
    total_train_step = 0
    # 测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 20
    # 神经网络识别总正确率
    total_accuracy = 0

    # ---------开始训练/测试---------
    # enumerate返回值结构展示
    print(list(enumerate(train_loader)))
    epoch = 50
    for e in range(epoch):
        # 设置为训练模型
        rn_module.train()
        print("-----第{}轮训练-----".format(e + 1))
        for enum, (imgs, targets) in enumerate(train_loader):
            # 启用GPU
            imgs = imgs.to(device)
            targets = targets.to(device)
            print("shape of imgs:", imgs.shape)
            print("shape of targets:", targets.shape)
            # 放入模型进行训练
            outputs = rn_module(imgs)
            loss = loss_func(outputs, targets.long())
            # 将优化器中的参数设置为0，每批图像进行优化前都要置零
            optimizer.zero_grad()
            # 用反向传播获得模型中参数的梯度
            loss.backward()
            # 调优
            optimizer.step()
            # 用于显示到tensorboard的训练步数
            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)

            # 测试步骤开始
            # 设置为测试模型
            rn_module.eval()
            # 将参数置零
            total_test_loss = 0
            total_accuracy = 0
            with torch.no_grad():
                for data in test_loader:
                    imgs, targets = data
                    # 启用GPU
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    outputs = rn_module(imgs)
                    loss = loss_func(outputs, targets.long())
                    # 测试集的总loss
                    total_test_loss += loss.item()
                    # outputs里是一个batch（64张图片）中对于各个图片的预测的数组，targets是正确答案，对比它们的差别就能得出正确率
                    total_accuracy += (outputs.argmax(1) == targets).sum()
            print("整体测试集上的Loss：{}".format(total_test_loss))
            # if total_accuracy / test_set_len == 0:
            #     print("total_accuracy:", total_accuracy)
            #     print("test_set_len:", test_set_len)
            print("整体测试集上的识别正确率：{}".format(total_accuracy / test_set_len))
            # 写入tensorboard展示
            writer.add_scalar("test_lost", total_test_loss, total_test_step)
            # 测试的步数
            total_test_step += 1
            # 保存模型，方式1
            torch.save(rn_module, "./models/ResNet_train_mode1_{}.pth".format(e))
            # 保存模型,方式2
            torch.save(rn_module.state_dict(), "./models/ResNet_train_mode2_{}.pth".format(e))
    writer.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test()
    main()
