import time
import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from torchvision import transforms

import utils

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"

def main():
    # model = torch.load("./result/pruned_vgg11_75.54_6.55s.pt")
    #
    # model.eval()
    onnxFile = "./result/pruned_vgg11_75.54_7.53s.onnx"
    # torch.onnx.export(model,
    #               torch.randn(1, 3, 32, 32, device="cuda"),
    #               onnxFile,
    #               input_names=["x"], #输入节点的名称列表
    #               output_names=["y"], #输出节点的名称列表
    #               do_constant_folding=True, #是否执行常量折叠优化
    #               verbose=True, #打印一些转换日志
    #               keep_initializers_as_inputs=True,
    #               opset_version=13,
    #               dynamic_axes={"x": {0: "nBatchSize"}} #设置动态的维度
    #               )
    # print("Succeeded converting model into ONNX!")

    # 执行onnxruntime测试
    onnx.checker.check_model(onnxFile) # 检查onnx模型
    if device == 'cpu':
        providers = ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnxFile,providers=providers)
    # x = np.random.randn(1, 3, 32, 32).astype(np.float32) # 模拟输入


    # 准备测试集
    train_path = "./dataset/"
    test_loader = utils.get_test_loader(train_path, 100) # 数字为batch_size
    test_time = 0.0
    correct = 0.0

    for (images, labels) in test_loader:
        images = images.numpy()
        start = time.time()
        outputs = session.run(None, {'x': images}) # 执行推理
        test_time += time.time()-start
        outputs = torch.tensor(outputs[0])
        preds = F.softmax(outputs, dim=1)
        preds = torch.argmax(preds, dim=1)
        # print(preds)
        correct += preds.eq(labels).sum()

    acc = (100. * correct) / len(test_loader.dataset)
    print('test top1_acc:{}'.format(acc))
    print('test time:{}s'.format(test_time))


if __name__ == '__main__':
    main()

    # ----------------------------------------------------
    # 以下为10张图片测试
    # model_path = "./result/pruned_vgg11_75.54_7.53s.onnx"
    # onnx.checker.check_model(model_path)  # 检查onnx模型
    # session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    #
    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                          (0.2023, 0.1994, 0.2010))
    # ])
    # testDataset = utils.MyData(transform=trans)
    # testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=10, shuffle=False)
    #
    # for (image, label) in testLoader:
    #     image = image.numpy()
    #     outputs = session.run(None, {'x': image}) # 执行推理
    #     outputs = torch.tensor(outputs[0])
    #     print(outputs)
    #     preds = F.softmax(outputs, dim=1)
    #     preds = torch.argmax(preds, dim=1)
    #     print(preds)


