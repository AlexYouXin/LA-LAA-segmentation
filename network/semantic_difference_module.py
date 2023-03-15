import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from conv_module import Conv3dReLU, Conv3dbn


# Semantic difference module
class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(SDC, self).__init__() 
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv3dbn(guidance_channels, in_channels, kernel_size=3, padding=1)                         
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # initialize for skipped features
        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        x_initial[:, :, np.int(kernel_size / 2), np.int(kernel_size / 2), np.int(kernel_size / 2)] = -1
        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, np.int(kernel_size / 2), np.int(kernel_size / 2), np.int(kernel_size / 2)].detach()

        # initialize for guidance features
        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        guidance_initial[:, :, np.int(kernel_size / 2), np.int(kernel_size / 2), np.int(kernel_size / 2)] = -1
        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, np.int(kernel_size / 2), np.int(kernel_size / 2), np.int(kernel_size / 2)].detach()


    def forward(self, x, guidance):
        guidance_channels = self.guidance_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size
        
        guidance = self.conv1(guidance)

        x_diff = F.conv3d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1, groups=in_channels)
        
        
        guidance_diff = F.conv3d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1, groups=in_channels)
        # print(self.guidance_kernel_diff[0, 0], self.x_kernel_diff[0, 0])
        out = self.conv(x_diff * guidance_diff * guidance_diff)
        return out, x_diff, guidance, guidance_diff
