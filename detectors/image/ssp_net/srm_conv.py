"""SRM conv from bcmi/SSP-AI-Generated-Image-Detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SRMConv2d_simple(nn.Module):
    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)
        return out

    def _build_kernel(self, inc):
        filter1 = [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ]
        filter2 = [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ]
        filter3 = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        filter1 = np.asarray(filter1, dtype=float) / 4.0
        filter2 = np.asarray(filter2, dtype=float) / 12.0
        filter3 = np.asarray(filter3, dtype=float) / 2.0
        filters = [[filter1], [filter2], [filter3]]
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)
        return filters
