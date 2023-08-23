import argparse
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

import cross_validate as train
from res_net import ResNet
import nii_dataset

# 选择使用gpu还是cpu进行训练
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(test_loader, model, criterion, total_test_step):
    writer = SummaryWriter("logs")
    # 设置为测试模型
    model.eval()
    test_pred = []
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(imgs)
            outputs = outputs.to(DEVICE)
            # 返回损失值
            loss = criterion(outputs, targets.long())
            # 准确率
            total_accuracy += (outputs.argmax(1) == targets).sum()
            # 将二分类预测结果整合起来
            test_pred.append(outputs.data.cpu().numpy())
            # 测试集Loss以及预测正确率
            # print("测试集Loss：{}".format(loss))
            # 写入tensorboard展示
            # writer.add_scalar("test_lost", loss, total_test_step)
    # writer.close()
    return loss, total_accuracy, np.vstack(test_pred)


def read_dataset(train_path, test_path, batch_size):
    """ 下载并创建训练集和测试集并装入dataloader """
    # 训练集dataloader
    train_loader = torch.utils.data.DataLoader(
        nii_dataset.NiiDataset(train_path,
                               A.Compose([  # 这些措施都是为了提供更多的数据，提高模型的识别能力
                                   A.RandomRotate90(),  # 随机旋转
                                   A.RandomCrop(120, 120),  # 随机裁剪图片
                                   A.HorizontalFlip(p=0.5),  # 水平翻转
                                   A.RandomContrast(p=0.5),
                                   A.RandomBrightnessContrast(p=0.5),
                               ])
                               ), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False
    )
    # 测试集dataloader
    test_loader = torch.utils.data.DataLoader(
        nii_dataset.NiiDataset(test_path,
                               A.Compose([
                                   A.RandomCrop(120, 120),
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


def generate():
    # ---------加载数据集---------
    # 数据集相关参数
    train_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Train\\*\\*")
    test_path = glob.glob("D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Test\\*")
    batch_size = 64
    total_test_step = 0
    label = None
    # 损失函数
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(DEVICE)

    # ---------读取数据---------
    train_dataloader, test_dataloader, train_set_len, test_set_len = read_dataset(train_path, test_path, batch_size)

    # ---------测试集增强，对于10个权重模型依次进行10轮预测，最后将结果相加---------
    pred = None
    for model_path in ['resnet18_fold0.pt', 'resnet18_fold1.pt', 'resnet18_fold2.pt',
                       'resnet18_fold3.pt', 'resnet18_fold4.pt', 'resnet18_fold5.pt',
                       'resnet18_fold6.pt', 'resnet18_fold7.pt', 'resnet18_fold8.pt',
                       'resnet18_fold9.pt']:
        model = ResNet()
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        for _ in range(10):
            test_loss, total_accuracy, test_pred = predict(test_dataloader, model, loss_func, total_test_step)
            if pred is None:
                pred = test_pred
            else:
                pred += test_pred
            label = pred.argmax(1)

    # 将label由索引转换为对应字符
    label = ['MCI' if x == 0 else 'NC' for x in label]

    # ---------生成提交csv文件---------
    file_name = "submit_test9.csv"
    submit(file_name, test_path, label)


def submit(file_name, test_path, label):
    submit_df = pd.DataFrame(
        {
            'uuid': [int(x.split('\\')[-1][:-4]) for x in test_path],
            'label': label
        })
    submit_df = submit_df.sort_values(by='uuid')
    # 展示文件内容
    print(submit_df)
    # 生成csv文件
    submit_df.to_csv(file_name, index=None)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str,
                        default="D:\\xuexi\\post-graduate\\py_projects\\ResNet-PET\\datasets\\Brain-PET\\Train\\*\\*",
                        help='dataset path')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--train-transforms', type=str, required=True, help='train transforms')
    # parser.add_argument('--val-transforms', type=str, required=True, help='val transforms')
    opt = parser.parse_args()
    return opt


def main(opt):
    # 进行训练以及交叉验证
    train.run(opt)
    # 对测试集进行预测并生成预测csv
    generate()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
