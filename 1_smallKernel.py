import time

import torch
from torch import nn
from torch.nn import functional as F

import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def main():

    # 1.导入pt权重文件
    print("----------------------------------------------------")
    print("1.print model:")

    model_path = './result/origin_vgg11_5C_65.35_2.674s.pt'  # vgg11
    net = torch.load(model_path)
    print(net)

    # 2.卷积核替换
    print("----------------------------------------------------")
    print("2.kernel modify:")
    for i,m in enumerate(net.feature):
        if isinstance(m, nn.Conv2d) and m.kernel_size==(5,5):
            net.feature[i] = nn.Sequential(
                nn.Conv2d(m.in_channels, m.out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(m.out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(m.out_channels, m.out_channels, kernel_size=3, padding=1, bias=False)
            )
    print(net)

    # 3.进行测试
    print("----------------------------------------------------")
    print("3.test:")
    net.cuda()
    train_path = './dataset/'
    test_loader = utils.get_test_loader(train_path)

    # utils.eval_training("minKer_vgg11_5C",net,test_loader,device)



if __name__ == '__main__':
    main()