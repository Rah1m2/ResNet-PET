# author: detoX
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from torch import nn

import res_net
from nii_dataset import NiiDataset
from train import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
skf = KFold(n_splits=10, random_state=233, shuffle=True)


def run(train_path):
    train_loss = None
    total_train_step = 0
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_path, train_path)):

        # train_idx = np.array([int(i) for i in train_idx])
        # train_idx = train_idx.astype(int)
        train_path = np.array(train_path)
        train_loader = torch.utils.data.DataLoader(
            NiiDataset(train_path[train_idx],
                       ...
                       ))

        val_loader = torch.utils.data.DataLoader(
            NiiDataset(train_path[val_idx],
                       ...
                       ))

        model = res_net.ResNet()
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), 0.001)

        for _ in range(3):
            train_loss, total_train_step = train(train_loader, model, criterion, optimizer, total_train_step)
            val_acc = validate(val_loader, model, criterion)
            train_acc = validate(train_loader, model, criterion)

            print("train_loss:", train_loss, "train_acc:", train_acc, "val_acc:", val_acc)
            torch.save(model.state_dict(), './resnet18_fold{0}.pt'.format(fold_idx))


def validate(val_loader, model, criterion):
    model.eval()

    val_acc = 0.0

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target.long())

            val_acc += (output.argmax(1) == target).sum().item()

    return val_acc / len(val_loader.dataset)
