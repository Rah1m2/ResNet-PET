# author: detoX
import glob
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import res_net
from nii_dataset import NiiDataset
from train import train
import albumentations as A

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
skf = KFold(n_splits=10, random_state=233, shuffle=True)


def run(opt):
    cross_validate(**vars(opt))


def cross_validate(train_path, batch_size, learning_rate):
    total_train_step = 0
    total_val_step = 0
    # 获取路径下所有图片的路径，并返回为列表形式
    train_path = glob.glob(train_path)
    # 训练集tran
    train_compose = A.Compose([  # 这些措施都是为了提供更多的数据，提高模型的识别能力
        A.RandomRotate90(),  # 随机旋转
        A.RandomCrop(120, 120),  # 随机裁剪图片
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.RandomContrast(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ])
    # 验证集tran
    val_compose = A.Compose([
        A.RandomCrop(120, 120),
    ])

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_path, train_path)):
        train_path = np.array(train_path)
        train_loader = torch.utils.data.DataLoader(
            NiiDataset(train_path[train_idx],
                       train_compose
                       ), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)

        val_loader = torch.utils.data.DataLoader(
            NiiDataset(train_path[val_idx],
                       val_compose
                       ), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)

        model = res_net.ResNet()
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

        for _ in range(3):
            train_loss, total_train_step = train(train_loader, model, criterion, optimizer, total_train_step)
            val_acc, total_val_step = validate(val_loader, model, criterion, total_val_step)
            train_acc = validate(train_loader, model, criterion, total_val_step)

            print("train_loss:", train_loss, "train_acc:", train_acc, "val_acc:", val_acc)
            torch.save(model.state_dict(), './resnet18_fold{0}.pt'.format(fold_idx))


def validate(val_loader, model, criterion, total_val_step):
    model.eval()
    writer = SummaryWriter("logs")
    val_acc = 0.0

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if input == "empty":
                continue
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target.long())
            total_val_step += 1
            if total_val_step % 10 == 0:
                writer.add_scalar("val_loss", loss, total_val_step)
            val_acc += (output.argmax(1) == target).sum().item()

    return val_acc / len(val_loader.dataset), total_val_step
