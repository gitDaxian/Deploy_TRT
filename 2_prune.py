# 进行适应性剪枝
import time
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

from utils import AverageMeter
import utils
from models.vgg import vgg

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
LossMeter = AverageMeter()
Top1AccMeter = AverageMeter()


# 对模型进行测试
def tst(model, test_loader, criterion):
    model.eval()
    LossMeter.reset()
    Top1AccMeter.reset()
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for img, label in test_loader:
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()
            logits = model(img)
            loss = criterion(logits, label)
            LossMeter.update(loss.item())
            pred = torch.nn.functional.softmax(logits, dim=1).argmax(dim=1)
            top1_acc = torch.eq(pred, label).sum().item() / len(img)
            Top1AccMeter.update(top1_acc)
        torch.cuda.synchronize()
        end = time.time()
        print("test loss:{}/ test top1_acc:{}".format(LossMeter.avg, Top1AccMeter.avg))
        print('test time:{}s'.format(end - start))


# 对模型进行一次剪枝
def prune(percent, model):  # 剪枝比例，总通道数，模型
    total = 0  # 获取总通道数
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    y, i = torch.sort(bn)
    thre_index = int(total * percent)  # 获取剪枝的阈值索引
    thre = y[thre_index]  # 获取剪枝的阈值
    # pruned = 0
    cfg = []  # 用于保存每个bn层要保留的通道数量
    cfg_mask = []  # 用于保存每个bn层的mask
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()  # 获取该bn层每个通道的gamma权重参数的绝对值
            mask = weight_copy.gt(thre).float().cuda()  # 获取该bn层的mask，其中1为要保留的通道，0为要剪枝的通道。
            # pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)  # 把要剪枝的通道权重归零
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))  # 保存该bn层要保留的通道数
            cfg_mask.append(mask.clone())  # 保存该bn层要保留的通道数
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    return cfg_mask, cfg


# trick:对BN层增加L1正则
def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.0001 * torch.sign(m.weight.data))  # L1


# 对模型进行微调
def finetune(model, train_loader, criterion, optimizer, cfg_mask):
    model.train()
    LossMeter.reset()
    Top1AccMeter.reset()
    for epoch in range(5):
        for img, label in train_loader:
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()
            optimizer.zero_grad()
            mask_index = 0
            # 不需要剪枝的通道不进行参数更新
            for k, m in enumerate(model.modules()):
                if isinstance(m, nn.BatchNorm2d):
                    mask = cfg_mask[mask_index]
                    m.weight.data.mul_(mask)
                    m.bias.data.mul_(mask)
                    mask_index += 1
                    img = m(img)
                elif isinstance(m, nn.Conv2d):
                    img = m(img)
                elif isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AvgPool2d):
                    img = m(img)
                elif isinstance(m, nn.Linear):
                    img = nn.AvgPool2d(2)(img)
                    img = img.view(img.size(0), -1)
                    img = m(img)
                elif isinstance(m, nn.ReLU):
                    img = m(img)
            # img = model(img)
            loss = criterion(img, label)
            pred = torch.nn.functional.softmax(img, dim=1).argmax(dim=1)
            loss.backward()
            updateBN(model)
            optimizer.step()
            LossMeter.update(loss.item())
            top1_acc = torch.eq(pred, label.to(device)).sum().item() / len(img)
            Top1AccMeter.update(top1_acc)
        print("finetune epoch:{} loss:{}/ top1_acc:{}".format(epoch, LossMeter.avg, Top1AccMeter.avg))


def get_new_model(model, newmodel, cfg_mask):
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # 获取非0通道对应的index
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            """把bn层权重转移到新的模型上"""
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))  # 获取非0输入通道对应的index
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # 获取非0输出通道对应的index
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            """把卷积层权重转移到新的模型上"""
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))  # 获取非0输入通道对应的index
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            """把全连接层权重转移到新的模型上"""
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

    return newmodel


def main():
    # 1.导入pt权重文件
    model_path = './result/kd_vgg11_83.88_12.215s.pt'  # vgg11
    pretrained_model = torch.load(model_path)

    print("----------------------------------------------------")
    print("1.print model:")
    print(pretrained_model)

    # 2.剪枝前先进行测试
    train_path = './dataset/'
    train_loader = utils.get_train_loader(train_path)
    test_loader = utils.get_test_loader(train_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(pretrained_model.parameters(), lr=0.001)
    print("----------------------------------------------------")
    print("2.start test before prune:")
    tst(pretrained_model, test_loader, criterion)

    # 3.进行迭代式剪枝操作
    print("----------------------------------------------------")
    print("3.iterate prune:")
    for percent in range(10, 60, 10):
        cfg_mask, cfg = prune(percent / 100.0, pretrained_model)
        print("finetune:{}/ prune percent:{}".format(percent / 10, percent / 100.0))
        finetune(pretrained_model, train_loader, criterion, optimizer, cfg_mask)
    # 4.获取新模型
    new_model = vgg(depth=11).cuda()
    # new_model = torch.load(model_path)
    new_model = get_new_model(pretrained_model, new_model, cfg_mask)
    print("----------------------------------------------------")
    print("4.print new_model:")
    print(new_model)
    # 5.剪枝后进行测试并保存权重
    print("----------------------------------------------------")
    print("5.start test after prune:")
    model_name = "pruned_vgg11"
    utils.eval_training(model_name,new_model, test_loader, device)


if __name__ == '__main__':
    # main()

    # ----------------------------------------------------
    # 以下为10张图片测试
    model_path = "./result/pruned_vgg11_75.54_6.55s.pt"
    net = torch.load(model_path).to(device)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    test_loader = utils.get_test_loader('./dataset/')
    # testDataset = utils.MyData(transform=trans)
    # testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=10, shuffle=False)
    infer_time = 0
    correct = 0.0
    for (image, label) in test_loader:
        image, label = image.to(device), label.to(device)
        with torch.no_grad(): #停止梯度计算
            start_time = time.time()
            outputs = net(image)
            infer_time += time.time() - start_time
            # print(outputs) # [100,10]
            preds = F.softmax(outputs, dim=1)
            preds = torch.argmax(preds, dim=1)
            correct += preds.eq(label).sum()
            # print(preds)

    acc = (100. * correct.float()) / len(test_loader.dataset)

    print("infer time:{}s".format(infer_time))
    print("test acc:{}".format(acc))
    # infer time:5.327565670013428s