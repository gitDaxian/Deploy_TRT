import time
import torch
import torch.nn as nn
import torchvision

from models.resnet import resnet34
from utils import AverageMeter
import utils

LossMeter = AverageMeter()
Top1AccMeter = AverageMeter()
num_epochs = 100
batch_size = 256

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader,criterion, optimizer, num_epochs):

    for epoch in range(num_epochs):
        model.train()
        LossMeter.reset()
        Top1AccMeter.reset()
        i=0
        for img, label in train_loader:
            logits = model(img.to(device))
            loss = criterion(logits, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LossMeter.update(loss.item())
            pred = torch.nn.functional.softmax(logits, dim=1).argmax(dim=1)
            top1_acc = torch.eq(pred, label.to(device)).sum().item() / len(img)
            Top1AccMeter.update(top1_acc)
            i+=1
            if i % 100 == 0:
                print("train epoch:{} loss:{}/ top1_acc:{}".format(epoch, LossMeter.avg, Top1AccMeter.avg))


def main():
    train_path = './dataset/'
    train_loader = utils.get_train_loader(train_path)
    test_loader = utils.get_test_loader(train_path)


    model = resnet34().cuda()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    # 训练模型
    train(model,train_loader,criterion,optimizer,num_epochs)
    # 测试并保存模型
    utils.eval_training("regnet",model,test_loader,device)


if __name__ == '__main__':
    main()

