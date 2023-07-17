import torch

if __name__ == '__main__':
    # 4.重新load测试模型
    new_model = torch.load('./result/minKer_vgg11_5C_10.0_2.66s.pt')
    print(new_model)