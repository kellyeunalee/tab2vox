import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3x3': lambda C, stride, affine: nn.AvgPool3d(kernel_size=3, stride=stride, padding=1),  
    'max_pool_3x3x3': lambda C, stride, affine: nn.MaxPool3d(kernel_size=3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine),  
    'conv_1x1x1': lambda C, stride, affine: conv3d(C, C, [1, 1, 1], stride=stride, padding=0, affine=affine),
    'conv_3x3x3': lambda C, stride, affine: conv3d(C, C, [3, 3, 3],  stride=stride, padding=1, affine=affine),
    'sep_conv_3x3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine), 
    'dil_conv_3x3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, affine),    
    'conv_5x5x5': lambda C, stride, affine: conv3d(C, C, [5, 5, 5],  stride=stride, padding=1, affine=affine),
    'conv_1x3x3': lambda C, stride, affine: VaniConv3d_Spatial_1x3x3(C, C, 3, stride, 1, affine=affine),
    'conv_3x1x1': lambda C, stride, affine: VaniConv3d_Temporal_3x1x1(C, C, 3, stride, 1, affine=affine),
}

class VaniConv3d_Spatial_1x3x3(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(VaniConv3d_Spatial_1x3x3, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_out, [1, 3, 3], stride=stride, padding=[0, 1, 1], bias=False),
            nn.BatchNorm3d(C_out, affine=affine)
        )
    def forward(self, x):
        return self.op(x)


class VaniConv3d_Temporal_3x1x1(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(VaniConv3d_Temporal_3x1x1, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_out, [3, 1, 1], stride=stride, padding=[1, 0, 0], bias=False),
            nn.BatchNorm3d(C_out, affine=affine)
        )
    def forward(self, x):
        return self.op(x)


def compute_pad(x):
    pad_dim2 = 0 if x.shape[2] % 2 == 0 else 1
    pad_dim3 = 0 if x.shape[3] % 2 == 0 else 1
    pad_dim4 = 0 if x.shape[4] % 2 == 0 else 1
    
    pad = (0, pad_dim4, 0, pad_dim3, 0, pad_dim2) 
    return pad
    
class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(C_in, C_out//2, kernel_size=1, stride=2, bias=False)
        self.conv2 = nn.Conv3d(C_in, C_out//2, kernel_size=1, stride=2, bias=False)
        self.bn = nn.BatchNorm3d(C_out, affine=affine) 

    def forward(self, x):
        x = self.relu(x)
        
        x1 = self.conv1(x)
        
        x_pad = F.pad(x, compute_pad(x))
        x2 = self.conv2(x_pad[:,:,1:,1:,1:]) 
        
        x = torch.cat([x1, x2], dim=1)   
        x = self.bn(x)
        return x


class Zero(nn.Module):  

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 2:
            x = x[:,:,::self.stride, ::self.stride, ::self.stride]
        return x.mul(0.)    


class conv3d(nn.Module):
    def __init__(self, in_C, out_C, kernel_size, stride, padding, affine=True):
        super(conv3d, self).__init__()
        self.ops = nn.Sequential(
            nn.Conv3d(in_C, out_C, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_C, affine=affine)
        )
    def forward(self, x):
        return self.ops(x)
    

class ReLUConvBN(nn.Module):
    def __init__(self, in_C, out_C, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.ops = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_C, out_C, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_C, affine=affine)
        )
    
    def forward(self, x):
        return self.ops(x)
    
    
class SepConv(nn.Module):
    def __init__(self, in_C, out_C, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.ops = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_C, in_C, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_C, bias=False),
            nn.Conv3d(in_C, in_C, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_C, affine=affine),
            nn.Conv3d(in_C, in_C, kernel_size=kernel_size, stride=1, padding=padding, groups=in_C, bias=False),
            nn.Conv3d(in_C, out_C, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_C, affine=affine)
        )
    
    def forward(self, x):
        return self.ops(x)
    
    
class DilConv(nn.Module):
    def __init__(self, in_C, out_C, kernel_size, stride, padding, affine=True):
        super(DilConv, self).__init__()
        self.ops = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_C, in_C, kernel_size=kernel_size, stride=stride, padding=padding, dilation=2, groups=in_C, bias=False),
            nn.Conv3d(in_C, out_C, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_C, affine=affine)
        )
    
    def forward(self, x):
        return self.ops(x)

