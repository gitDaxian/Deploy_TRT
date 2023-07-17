import time

import numpy as np
import onnx
import onnxoptimizer as optimizer
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torchvision import transforms

import utils


def main():
    # Preprocessing: load the model contains two transposes.
    # model_path = os.path.join('resources', 'two_transposes.onnx')
    # original_model = onnx.load(model_path)
    original_model = onnx.load("./result/pruned_vgg11_75.54_7.53s.onnx")
    # Check that the IR is well formed
    onnx.checker.check_model(original_model)
    print('The model before optimization:\n\n{}'.format(onnx.helper.printable_graph(original_model.graph)))


    # A full list of supported optimization passes can be found using get_available_passes()
    all_passes = optimizer.get_available_passes()
    print("Available optimization passes:")
    for p in all_passes:
        print('\t{}'.format(p))
    print()

    # Pick one pass as example
    passes = ['extract_constant_to_initializer']
    # passes = ['fuse_bn_into_conv']
    # Apply the optimization on the original serialized model
    optimized_model = optimizer.optimize(original_model, passes)

    print('The model after optimization:\n\n{}'.format(onnx.helper.printable_graph(optimized_model.graph)))
    # save new model
    onnx.save(optimized_model, "./result/onnxopted_vgg11_75.54_7.53s.onnx")

    optimized_model_path = "./result/onnxopted_vgg11_75.54_7.53s.onnx"
    # 执行onnxruntime测试
    onnx.checker.check_model(optimized_model_path) # 检查onnx模型
    session = ort.InferenceSession(optimized_model_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # 准备测试集
    train_path = "./dataset/"
    test_loader = utils.get_test_loader(train_path, 100) # 数字为batch_size
    correct = 0.0
    start = time.time() # 开始测试时间

    for (images, labels) in test_loader:
        images = images.numpy()
        outputs = session.run(None, {'x': images}) # 执行推理

        outputs = torch.tensor(outputs[0])
        preds = F.softmax(outputs, dim=1)
        preds = torch.argmax(preds, dim=1)
        print(preds)
        correct += preds.eq(labels).sum()

    end = time.time() # 结束测试时间
    acc = (100. * correct) / len(test_loader.dataset)
    test_time = end-start
    print('test top1_acc:{}'.format(acc))
    print('test time:{}s'.format(test_time))

    # 保留三位有效数字
    acc = round(acc.item(),3)
    test_time = round(test_time,3)

    optimized_model = onnx.load(optimized_model_path)
    file = "./result/onnxopted_vgg11_"+str(acc)+"_"+str(test_time)+"s.onnx"
    onnx.save(optimized_model,file)


if __name__ == '__main__':
    main()

    # ----------------------------------------------------
    # 以下为10张图片测试
    # model_path = "./result/onnxopted_vgg11_75.54_8.379s.onnx"
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