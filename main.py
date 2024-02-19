# -*-coding:utf-8 -*-
"""
# File       : test2.py
# Time       ：2024/2/19 14:39
# Author     ：wcirq
# version    ：python 3
# Description：
"""

import torch
import torch.torch_version

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def round(x, decimals=2):
    return torch.round(x*10**decimals)/10**decimals


def main():
    x = torch.rand(3, 3, dtype=torch.float32)
    x = torch.arange(-50., 50, 2)
    scale = 0.2
    zero_point = 10
    xq = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=torch.qint8)

    # ---torch.quantize_per_tensor 相当于-------
    if xq.dtype==torch.quint8:
        range_min = 0
        range_max = 255
    elif xq.dtype==torch.qint8:
        range_min = -128
        range_max = 127
    clip_min = (range_min-zero_point)*scale
    clip_max = (range_max-zero_point)*scale
    xq_self = torch.clip(x, clip_min, clip_max)
    xq_self = round(xq_self, decimals=1)
    # -----------------------------------------

    xq2 = xq.int_repr()

    # int_repr 相当于
    xq2_self = (xq_self / scale + zero_point).int()

    x_re = xq.dequantize()
    print()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
