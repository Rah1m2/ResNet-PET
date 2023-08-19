# author: detoX
import torch
from torch.utils.tensorboard import SummaryWriter

# 选择使用gpu还是cpu进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def run(opt):
    train(**vars(opt))

def train(train_loader, model, criterion, optimizer, total_train_step):
    # 总loss
    total_loss = 0
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
        total_loss += loss.item()
        if total_train_step % 10 == 0:
            print("训练次数：{}，Loss：{}, total_loss: {}".format(total_train_step, loss.item(), total_loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    return total_loss/len(train_loader), total_train_step