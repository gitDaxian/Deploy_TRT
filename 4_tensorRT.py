import time

import torch
import torch.nn.functional as F
from cuda import cudart
import cv2
import numpy as np
import os
import tensorrt as trt
from torchvision import transforms

import utils
import calibrator

# for FP16 mode
bUseFP16Mode = True
# for INT8 model
bUseINT8Mode = False

onnxFile = "./result/pruned_vgg11_76.29_4.258s_opt.onnx"
trtFile = "./result/model.plan"
inferenceImage = "./dataset/dog4.png"

nHeight = 32
nWidth = 32


def main():
    # TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
    if bUseFP16Mode:
        config.set_flag(trt.BuilderFlag.FP16)
    if bUseINT8Mode:
        config.set_flag(trt.BuilderFlag.INT8)
        # config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth), cacheFile)

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnxFile, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, (1, 3, nHeight, nWidth), (4, 3, nHeight, nWidth), (200, 3, nHeight, nWidth))
    config.add_optimization_profile(profile)

    # network.unmark_output(network.get_output(0))  # 去掉输出张量 "y"
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, "wb") as f:
        f.write(engineString)

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0, [100, 3, nHeight, nWidth])
    #print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(nInput):
       print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput, nInput + nOutput):
       print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    # data = cv2.imread(inferenceImage).astype(np.float32).reshape(1, 3, nHeight, nWidth)
    # data = np.range(10 * 3 * 32 * 32, dtype=np.float32).reshape(10, 3, 32, 32)

    # 准备测试集
    train_path = "./dataset/"
    test_loader = utils.get_test_loader(train_path, 100) # 数字为batch_size
    correct = 0.0
    # start = time.time() # 开始测试时间
    infer_time = 0

    for (images, labels) in test_loader:
        images = images.numpy()

        execute_start = time.time()

        bufferH = []
        bufferH.append(images)

        for i in range(nOutput):
            bufferH.append(
                np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
        bufferD = []
        for i in range(engine.num_bindings):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        context.execute_v2(bufferD)
        for i in range(nOutput):
            cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        for buffer in bufferD:
            cudart.cudaFree(buffer)
        # print("inputH0 :", bufferH[0].shape)
        # print("outputH0:", bufferH[-1].shape)
        # print(bufferH[-1])
        execute_end = time.time()
        infer_time += execute_end - execute_start

        bufferH[-1] = torch.tensor(bufferH[-1])
        preds = F.softmax(bufferH[-1], dim=1)
        preds = torch.argmax(preds, dim=1)
        # print(preds)
        correct += preds.eq(labels).sum()

    # end = time.time()  # 结束测试时间
    print("infer time:{}s".format(infer_time))
    # bs:
    # infer time:0.9842698574066162s bs:100
    # infer time:0.7969822883605957s bs:200
    # infer time:0.8126277923583984s bs:400 ERR:out of memory

    # infer time:0.7983741760253906s

    acc = (100. * correct) / len(test_loader.dataset)
    # test_time = end-start
    print('test top1_acc:{}'.format(acc))
    # print('test time:{}s'.format(test_time))
    # test time:6.766223192214966s bs:100
    # test time:7.0331339836120605s bs:200
    # test time:6.9211413860321045s bs:400

    print("Succeeded running model in TensorRT!")

    # ----------------------------------------------------
    # 以下为10张图片测试
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
    #     # test_end = time.time()
    #     # print("trans time:{}".format(test_end-test_start))
    #     bufferH = []
    #     bufferH.append(image)
    #
    #     for i in range(nOutput):
    #         bufferH.append(
    #             np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    #     bufferD = []
    #     for i in range(engine.num_bindings):
    #         bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
    #
    #     for i in range(nInput):
    #         cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes,
    #                           cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    #
    #     context.execute_v2(bufferD)
    #
    #     for i in range(nOutput):
    #         cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes,
    #                           cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    #
    #     # print("inputH0 :", bufferH[0].shape)
    #     # print("outputH0:", bufferH[-1].shape)
    #     print(bufferH[-1])
    #     bufferH[-1] = torch.tensor(bufferH[-1])
    #     preds = F.softmax(bufferH[-1], dim=1)
    #     preds = torch.argmax(preds, dim=1)
    #     print(preds)
    #     for buffer in bufferD:
    #         cudart.cudaFree(buffer)

if __name__ == '__main__':
    main()

