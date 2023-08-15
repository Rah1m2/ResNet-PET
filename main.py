# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import time

import pandas as pd
import torch
import glob
from torch import nn
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from res_net import ResNet
import nii_dataset


# class ResNet(nn.Module):
#     def __init__(self):
#         super(ResNet, self).__init__()
#
#         model = models.resnet18(True)
#         model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         model.avgpool = nn.AdaptiveAvgPool2d(1)
#         model.fc = nn.Linear(512, 2)
#         self.resnet = model
#
#     def forward(self, img):
#         out = self.resnet(img)
#         return out


# 选择使用gpu还是cpu进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_loader, model, criterion, optimizer, total_train_step):
    # 总loss
    loss = None
    # 训练步数
    train_step = 0
    # 设置为训练模型
    model.train()
    for enum, (imgs, targets) in enumerate(train_loader):
        writer = SummaryWriter("logs")
        # 启用GPU
        imgs = imgs.to(device)
        targets = targets.to(device)
        # 放入模型进行训练
        outputs = model(imgs)
        loss = criterion(outputs, targets.long())
        # 将优化器中的参数设置为0，每批图像进行优化前都要置零
        optimizer.zero_grad()
        # 用反向传播获得模型中参数的梯度
        loss.backward()
        # 调优
        optimizer.step()
        total_train_step += 1
        if total_train_step % 10 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    return loss, total_train_step


def validate(val_loader, model, criterion):
    model.eval()

    val_acc = 0.0

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target.long())

            val_acc += (output.argmax(1) == target).sum().item()

    return val_acc / len(val_loader.dataset)


def predict(test_loader, model, criterion, total_test_step):
    writer = SummaryWriter("logs")
    # 设置为测试模型
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            # 启用GPU
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets.long())
            total_accuracy += (outputs.argmax(1) == targets).sum()
            # 测试集Loss以及预测正确率
        print("测试集Loss：{}".format(loss))
        # 写入tensorboard展示
        writer.add_scalar("test_lost", loss, total_test_step)
        writer.close()
    return loss, total_accuracy


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


def verification():
    # ---------初始化---------
    # 神经网络模型
    ln_model = ResNet()
    # 获取保存的参数（没有网络结构）
    state_dict = torch.load("./models/ResNet_train_mode2_29.pth")
    # 将参数加载到网络模型中
    ln_model.load_state_dict(state_dict)
    # 加载数据集
    # 数据集相关参数
    train_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Train\\*\\*")
    test_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Test\\*")
    batch_size = 64
    # 读取数据
    train_dataloader, test_dataloader, train_set_len, test_set_len = read_dataset(train_path, test_path, batch_size)
    print(test_dataloader)
    # 初始化tensorboard
    # writer = SummaryWriter("tensor_img_logs")
    complete_res = None
    # ---------循环读取batch---------
    batch_count = 0
    for test_data in test_dataloader:
        imgs, targets = test_data
        print("shape of imgs:", imgs.shape)
        print("shape of targets:", targets.shape)
        # 设置为测试模式
        ln_model.eval()
        # 不计算梯度，节省性能
        with torch.no_grad():
            res = ln_model(imgs)
        accuracy = (res.argmax(1) == targets).sum()
        # 将完整数据的label拼接为数组
        if complete_res is None:
            complete_res = res.argmax(1).numpy()
        else:
            complete_res = np.append(complete_res, res.argmax(1).numpy())
        # ---------输出验证结果---------
        print("result:\n", res)
        # print("targets:\n", targets)
        print("prediction:\n", res.argmax(1))
        print("result comparison:\n", res.argmax(1) == targets)
        # print("accuracy of this batch:\n", (accuracy / batch_size).item())
        # writer.add_images("script_batch_{}".format(batch_count), imgs)
        batch_count += 1
    # writer.close()

    label = ['MCI' if x == 0 else 'NC' for x in complete_res]
    # 生成提交csv文件
    submit(test_path, label)


def read_dataset(train_path, test_path, batch_size):
    """ 下载并创建训练集和测试集并装入dataloader """
    # 训练集dataloader
    train_loader = torch.utils.data.DataLoader(
        nii_dataset.NiiDataset(train_path,
                               A.Compose([
                                   A.RandomRotate90(),
                                   A.RandomCrop(120, 120),
                                   A.HorizontalFlip(p=0.5),
                                   A.RandomContrast(p=0.5),
                                   A.RandomBrightnessContrast(p=0.5),
                               ])
                               ), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False
    )
    # 测试集dataloader
    test_loader = torch.utils.data.DataLoader(
        nii_dataset.NiiDataset(test_path,
                               A.Compose([
                                   A.RandomCrop(128, 128),
                                   A.HorizontalFlip(p=0.5),
                                   A.RandomContrast(p=0.5),
                               ])
                               ), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False
    )
    # len(train_loader)得出的结果是batch的数量，必须乘以batch_size才是数据的总数
    train_set_len = len(train_loader) * batch_size
    test_set_len = len(test_loader) * batch_size
    # 数据集长度
    print("训练数据集长度：{}".format(train_set_len))
    print("测试数据集长度：{}".format(test_set_len))
    return train_loader, test_loader, train_set_len, test_set_len


def submit(test_path, label):
    submit_df = pd.DataFrame(
        {
            'uuid': [int(x.split('\\')[-1][:-4]) for x in test_path],
            'label': label
        })
    submit_df = submit_df.sort_values(by='uuid')
    # 展示文件内容
    print(submit_df)
    # 生成csv文件
    submit_df.to_csv('submit.csv', index=None)


def main():
    # ---------初始化tensorboard---------
    writer = SummaryWriter("logs")

    # ---------加载数据集---------
    # 数据集相关参数
    train_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Train\\*\\*")
    test_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Test\\*")
    batch_size = 4
    # 随机打乱
    np.random.shuffle(train_path)
    np.random.shuffle(test_path)
    # 返回数据集参数
    train_loader, test_loader, train_set_len, test_set_len = read_dataset(train_path, test_path, batch_size)

    # ---------初始化网络---------
    # 神经网络模型
    rn_model = ResNet()
    rn_model = rn_model.to(device)
    # 损失函数，使用GPU进行训练
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)
    # 优化器
    learning_rate = 0.001
    # optimizer = torch.optim.SGD(rn_model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(rn_model.parameters(), lr=learning_rate)

    # ---------初始化训练相关参数---------
    # 训练次数
    total_train_step = 0
    # 测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 30
    # 神经网络识别总正确率
    total_accuracy = 0

    # ---------训练/测试---------
    for e in range(epoch):
        # ---------训练步骤---------
        print("-----第{}轮训练-----".format(e + 1))
        train_loss, total_train_step = train(train_loader, rn_model, loss_func, optimizer, total_train_step)
        # ---------测试步骤---------
        print("-----第{}轮测试-----".format(e + 1))
        test_loss, total_accuracy = predict(test_loader, rn_model, loss_func, total_test_step)
        # 测试集预测正确率
        print("整体测试集上的识别正确率：{}".format(total_accuracy / test_set_len))
        # 步数+1
        total_train_step += 1
        total_test_step += 1
        # 保存模型，方式1
        torch.save(rn_model, "./models/ResNet_train_mode1_{}.pth".format(e))
        # 保存模型,方式2
        torch.save(rn_model.state_dict(), "./models/ResNet_train_mode2_{}.pth".format(e))

    writer.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test()
    # main()
    verification()
