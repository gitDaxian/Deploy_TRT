import time

from glob import glob
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 统计loss和top1Acc
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

# 获取训练集train_loader
def get_train_loader(train_path, batch_size=256, num_workers=4):
    train_dataset = torchvision.datasets.CIFAR100(root=train_path,
                                                 train=True,
                                                 download=False,
                                                 transform=transforms.Compose([
                                                     transforms.Pad(4),
                                                     transforms.RandomCrop(32),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                          (0.2023, 0.1994, 0.2010))
                                                 ]))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    return train_loader

# 获取测试集test_loader
def get_test_loader(test_path, batch_size=256, num_workers=4):
    test_dataset = torchvision.datasets.CIFAR100(root=test_path,
                                                 train=False,
                                                 download=False,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                          (0.2023, 0.1994, 0.2010))
                                                 ]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers)
    return test_loader

# 测试网络
def eval_training(model_name,net,test_loader,device):
    net.eval() #作用之一：停止dropout层生效
    test_loss = 0.0
    correct = 0.0
    # 定义交叉熵损失函数
    lossCE = nn.CrossEntropyLoss()
    infer_time = 0.0
    for (images, labels) in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad(): #停止梯度计算
            start = time.time()
            outputs = net(images)
            infer_time += time.time() - start
            loss = lossCE(outputs, labels).to(device)
            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

    loss = test_loss / len(test_loader.dataset)
    acc = (100. * correct.float()) / len(test_loader.dataset)
    print('test loss:{}, top1_acc:{}'.format(loss,acc))
    print('test time:{}s'.format(infer_time))

    # 保留三位有效数字
    acc = round(acc.item(),3)
    test_time = round(infer_time,3)

    file = "result/" + model_name + "_" + str(acc) + "_"+ str(test_time) +"s.pt"
    torch.save(net, file)


class MyData(torch.utils.data.Dataset):

    def __init__(self,transform = None):
        self.data = glob("./dataset/test_data/*.png")
        self.transform = transform

    def __getitem__(self, index):
        imageName = self.data[index]
        data = cv2.imread(imageName)
        if self.transform is not None:
            data = self.transform(data)
        label = np.zeros(10, dtype=np.float32)
        index = int(imageName[-5])
        label[index] = 1
        return data, torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
