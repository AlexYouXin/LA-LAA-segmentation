import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from conv_module import Conv3dReLU, Conv3dbn
from semantic_difference_module import SDC


class SDN(nn.Module):
    def __init__(self, in_channel=3, guidance_channels=2):
        super(SDN, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(in_channel)

    def forward(self, feature, guidance):
        boundary_enhanced, x_diff, guidance, guidance_diff = self.sdc1(feature, guidance)
        boundary = self.relu(self.bn(boundary_enhanced))
        boundary_enhanced = boundary + feature          
        return boundary_enhanced
        



class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        channels = [16, 32, 64, 128, 256, 512]
        self.encoder1 = nn.Sequential(
            Conv3dReLU(in_channel, channels[0], kernel_size=3, padding=1),
            Conv3dReLU(channels[0], channels[1], kernel_size=3, padding=1)
        )                                                            # b, 16, 10, 10
        self.encoder2 = nn.Sequential(
            Conv3dReLU(channels[1], channels[1], kernel_size=3, padding=1),
            Conv3dReLU(channels[1], channels[2], kernel_size=3, padding=1)
        )    
        self.encoder3 = nn.Sequential(
            Conv3dReLU(channels[2], channels[2], kernel_size=3, padding=1),
            Conv3dReLU(channels[2], channels[3], kernel_size=3, padding=1)
        )
        self.encoder4 = nn.Sequential(
            Conv3dReLU(channels[3], channels[3], kernel_size=3, padding=1),
            Conv3dReLU(channels[3], channels[4], kernel_size=3, padding=1)
        )
        self.encoder5 = nn.Sequential(
            Conv3dReLU(channels[4], channels[4], kernel_size=3, padding=1),
            Conv3dReLU(channels[4], channels[5], kernel_size=3, padding=1)
        )
        
        self.decoder1 = nn.Sequential(
            Conv3dReLU(channels[5] + channels[4], channels[4], kernel_size=3, padding=1),
            Conv3dReLU(channels[4], channels[4], kernel_size=3, padding=1)
        )
        self.decoder2 = nn.Sequential(
            Conv3dReLU(channels[4] + channels[3], channels[3], kernel_size=3, padding=1),
            Conv3dReLU(channels[3], channels[3], kernel_size=3, padding=1)
        )
        self.decoder3 = nn.Sequential(
            Conv3dReLU(channels[3] + channels[2], channels[2], kernel_size=3, padding=1),
            Conv3dReLU(channels[2], channels[2], kernel_size=3, padding=1)
        )  # b, 1, 28, 28
        self.decoder4 = nn.Sequential(
            Conv3dReLU(channels[2] + channels[1], channels[1], kernel_size=3, padding=1),
            Conv3dReLU(channels[1], channels[1], kernel_size=3, padding=1)
        )
        
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.segmentation_head = nn.Conv3d(channels[1], out_channel, kernel_size=3, padding=1)

        self.sdn1 = SDN(channels[4], channels[5])
        self.sdn2 = SDN(channels[3], channels[4])
        self.sdn3 = SDN(channels[2], channels[3])


    def forward(self, x):
        # print(x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1,1)
            
        out = self.encoder1(x)
        t1 = out
        out = self.down(out)
        
        
        out = self.encoder2(out)
        t2 = out
        out = self.down(out)

        
        out = self.encoder3(out)
        t3 = out
        out = self.down(out)

        
        out = self.encoder4(out)
        t4 = out

        out = self.down(out)

        
        out = self.encoder5(out)
        

        out = self.up(out)
        t4 = self.sdn1(t4, out)
        out = torch.cat((out, t4), 1)        # /8
        out = self.decoder1(out)

        
        out = self.up(out)
        t3 = self.sdn2(t3, out)
        out = torch.cat((out, t3), 1)        # /4
        out = self.decoder2(out)

        
        out = self.up(out)
        t2 = self.sdn3(t2, out)
        out = torch.cat((out, t2), 1)         # /2
        out = self.decoder3(out)
        
        out = self.up(out)
        out = torch.cat((out, t1), 1)     # /1
        out = self.decoder4(out)
        
        out = self.segmentation_head(out)
        

        return out