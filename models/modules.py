

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.odconv import ODConv2d

def odconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, kernels_num=4):   
    return nn.Sequential(
            ODConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation,reduction=0.0625, kernel_num=kernels_num),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True))

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1,inplace=True))

def conv_leakyrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=True),
        nn.LeakyReLU(0.1)
    )

def convtranspose_leakyrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True),
        nn.LeakyReLU(0.1)
    )



class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


def depth_regression(prob_volume, depth_values):
    if not len(depth_values.shape) == len(prob_volume.shape):
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(prob_volume * depth_values, 1)
    return depth


def winner_take_all(prob_volume, depth_values):
    _, idx = torch.max(prob_volume, dim=1, keepdim=True)
    depth = torch.gather(depth_values, 1, idx).squeeze(1)
    return depth


def calInitDepthHypos(init_depth_hypos, shape):
    depth_min, depth_max = init_depth_hypos["depth_min"], init_depth_hypos["depth_max"]
    n_depths = init_depth_hypos["n_depths"]
    dtype, device = depth_min.dtype, depth_min.device

    interval = (depth_max - depth_min) / (n_depths.float() - 1)
    depth_values = depth_min.unsqueeze(1) + (torch.arange(0, n_depths[0], device=device, dtype=dtype, requires_grad=False).reshape(1, -1) * interval.unsqueeze(1))
    depth_values = depth_values.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[2], shape[3])

    return depth_values


def calDeptyHypos(depth_up, init_depth_hypos, n_depths, ratio):
    depth_interval = init_depth_hypos["depth_interval"][0]
    dtype, device = depth_interval.dtype, depth_interval.device

    depth_min = (depth_up - n_depths / 2 * depth_interval * ratio)      # depth_interval = 2.5*1.06
    depth_max = (depth_up + n_depths / 2 * depth_interval * ratio)

    interval = (depth_max - depth_min) / (n_depths - 1)
    depth_values = depth_min.unsqueeze(1) + (torch.arange(0, n_depths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1, 1, 1) * interval.unsqueeze(1))

    return depth_values
