# -*-coding:utf-8 -*-
"""
# File       : test2.py
# Time       ：2024/2/19 14:39
# Author     ：wcirq
# version    ：python 3
# Description：静态量化
# Link       ：https://zhuanlan.zhihu.com/p/299108528
"""
import time
import torch
from torch import nn


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(3, 32, 3)
        layers = [torch.nn.Conv2d(32, 32, 3) if i % 2 == 0 else torch.nn.ReLU() for i in range(10)]
        self.sequence = nn.Sequential(
            *layers
        )

        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.conv(x)
        x = self.sequence(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


if __name__ == '__main__':
    model_fp32 = Model()
    model_fp32.eval()
    # qnnpack是基于tensor的模式，这种相当于一种全局量化模式；另外fbgemm是基于通道的模式，相比qnnpack，fbgemm是以及基于通道的局部量化模式；需要说明的是这两种模式主要是针对权重而言的，激活函数的量化都是一样的。
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [[f'sequence.{i*2}', f'sequence.{i*2+1}'] for i in range(len(model_fp32.sequence)//2)])
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

    # 喂数据，计算数据分布
    for i in range(2):
        datas = torch.randn(4, 3, 128, 128)
        model_fp32_prepared(datas)

    model_int8 = torch.quantization.convert(model_fp32_prepared)

    input_fp32 = torch.randn(4, 3, 128, 128)
    # run the model, relevant calculations will happen in int8
    res_fp32 = model_fp32(input_fp32)
    res_int8 = model_int8(input_fp32)

    print(model_fp32)
    print(model_int8)

    s = time.time()
    for i in range(5):
        model_fp32(input_fp32)
    print("model_fp32:", time.time() - s)

    s = time.time()
    for i in range(5):
        model_int8(input_fp32)
    print("model_int8:", time.time() - s)

    s = time.time()
    for i in range(30):
        model_fp32(input_fp32)
    print("model_fp32:", time.time() - s)

    s = time.time()
    for i in range(30):
        model_int8(input_fp32)
    print("model_int8:", time.time() - s)
