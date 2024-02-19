# -*-coding:utf-8 -*-
"""
# File       : test2.py
# Time       ：2024/2/19 14:39
# Author     ：wcirq
# version    ：python 3
# Description：动态量化
"""
import time
import torch
from torch import nn


class CivilNet(nn.Module):
    def __init__(self):
        super(CivilNet, self).__init__()
        gemfieldin = 1
        self.gemfieldout = 3
        self.conv = nn.Conv2d(gemfieldin, self.gemfieldout, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.fc = nn.Linear(1024, 3, bias=False)
        self.sequence = nn.Sequential(
                        nn.Linear(3, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True),
                        nn.Linear(1024, 1024, bias=True)
                       )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], self.gemfieldout, -1)
        x = self.fc(x)
        x = self.sequence(x)
        x = self.relu(x)
        return x


if __name__ == '__main__':
    model_fp32 = CivilNet()
    model_fp32.eval()
    model_int8 = torch.quantization.quantize_dynamic(model_fp32, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False)

    input_fp32 = torch.rand((32, 1, 32, 32))
    model_fp32(input_fp32)
    model_int8(input_fp32)

    s = time.time()
    for i in range(30):
        model_fp32(input_fp32)
    print("model_fp32:", time.time()-s)

    s = time.time()
    for i in range(30):
        model_int8(input_fp32)
    print("model_int8:", time.time()-s)

    s = time.time()
    for i in range(30):
        model_fp32(input_fp32)
    print("model_fp32:", time.time()-s)

    s = time.time()
    for i in range(30):
        model_int8(input_fp32)
    print("model_int8:", time.time()-s)
