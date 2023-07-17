import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from models.vgg import vgg
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LossMeter = utils.AverageMeter()
Top1AccMeter = utils.AverageMeter()

def train(stu_net, tea_net, train_loader, epochs, device):
    # 蒸馏温度 系数
    T, lambda_stu = 4.0, 0.3
    optimizer = optim.Adam(stu_net.parameters(), lr=0.0001)
    lossKD = nn.KLDivLoss()
    lossCE = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        tea_net.eval()  # = model.train(mode=False)
        stu_net.train()
        LossMeter.reset()
        Top1AccMeter.reset()
        for batch_index, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            # 1
            y_student = stu_net(image)
            # 2
            y_teacher = tea_net(image)
            y_teacher = y_teacher.detach()  # 【特别注意】切断老师网络的反向传播

            # 3
            loss_dis = lossKD(F.log_softmax(y_student / T, dim=1),
                              F.softmax(y_teacher / T, dim=1))
            # 4 5
            loss_stu = lossCE(y_student, label)

            loss = (1 - lambda_stu) * T * T * loss_dis + lambda_stu * loss_stu
            loss = loss.to(device)
            loss.backward()
            optimizer.step()

            LossMeter.update(loss.item())
            pred = torch.nn.functional.softmax(y_student, dim=1).argmax(dim=1)
            top1_acc = torch.eq(pred, label.to(device)).sum().item() / len(image)
            Top1AccMeter.update(top1_acc)
            if batch_index % 100 == 0:
                print("train epoch:{} loss:{}/ top1_acc:{}".format(epoch, LossMeter.avg, Top1AccMeter.avg))


def kd(tea_net, stu_net, epochs=20):
    batch_size = 128

    train_path = "./dataset/"
    train_loader = utils.get_train_loader(train_path, batch_size)
    test_loader = utils.get_test_loader(train_path, batch_size)

    # train(stu_net, tea_net, train_loader, epochs, device)

    stu_net = torch.load("./result/kd_vgg11_83.88_12.215s.pt")

    # 测试并保存
    model_name = "kd_vgg11"
    utils.eval_training(model_name,stu_net, test_loader, device)


if __name__ == "__main__":
    # # 1.指定训练好的教师模型
    # tea_path = "./result/p_vgg19.pt"
    # tea_net = torch.load(tea_path)
    # tea_net = tea_net.cuda()
    #
    # # 2.指定学生模型
    # stu_net = vgg(depth=11)
    # stu_net = stu_net.cuda()
    #
    # # 3.进行蒸馏
    # kd(tea_net, stu_net)

    # ----------------------------------------------------
    # 以下为10张图片测试
    net = torch.load("./result/kd_vgg11_83.88_12.215s.pt")

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    testDataset = utils.MyData(transform=trans)
    testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=10, shuffle=False)

    for (image, label) in testLoader:
        image, label = image.to(device), label.to(device)
        with torch.no_grad(): #停止梯度计算
            outputs = net(image)
            print(outputs)
            preds = F.softmax(outputs, dim=1)
            preds = torch.argmax(preds, dim=1)
            print(preds)
