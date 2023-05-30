import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            dilation=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)

class Conv3dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            dilation=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)



class network(nn.Module):
    def __init__(self, in_channel=2, out_channel=3):
        super(network, self).__init__()
        hidden_dim = 16
        self.conv1 = Conv3dReLU(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1, stride=2, dilation=1)
        self.conv2 = Conv3dReLU(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1, stride=2, dilation=1)
        self.conv3 = Conv3dReLU(hidden_dim * 4, hidden_dim * 8, kernel_size=3, padding=1, stride=2, dilation=1)
        self.conv4 = Conv3dReLU(hidden_dim * 8, hidden_dim * 8, kernel_size=3, padding=1, stride=1, dilation=1)
        
        self.conv = Conv3dbn(in_channel, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.segmentation_head = nn.Conv3d(hidden_dim, out_channel, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.de_conv1 = Conv3dReLU(hidden_dim * 16, hidden_dim * 4, kernel_size=3, padding=1, stride=1, dilation=1)
        self.de_conv2 = Conv3dReLU(hidden_dim * 8, hidden_dim * 2, kernel_size=3, padding=1, stride=1, dilation=1)
        self.de_conv3 = Conv3dReLU(hidden_dim * 4, hidden_dim * 1, kernel_size=3, padding=1, stride=1, dilation=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.conv1(x)
        skip1 = x
        
        x = self.conv2(x)
        skip2 = x
        
        x = self.conv3(x)
        skip3 = x
        
        x = self.conv4(x)
        
        # decoder
        x = torch.cat((x, skip3), 1)
        x = self.de_conv1(x)
        
        x = torch.cat((self.up(x), skip2), 1)
        x = self.de_conv2(x)
        
        x = torch.cat((self.up(x), skip1), 1)
        x = self.de_conv3(x)

        x = self.up(x)
        x = self.segmentation_head(x)
        return x
        

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
        )     # b, 8, 3, 3
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

    def forward(self, x):
        # print(x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1,1)
            
        out = self.encoder1(x)
        t1 = out
        out = self.down(out)
        
        # print(out.shape)
        
        out = self.encoder2(out)
        t2 = out
        out = self.down(out)
        
        # print(out.shape)
        
        out = self.encoder3(out)
        t3 = out
        out = self.down(out)
        
        # print(out.shape)
        
        out = self.encoder4(out)
        t4 = out
        # print('t4 shape: ', t4.shape)
        out = self.down(out)
        
        # print(out.shape)
        
        out = self.encoder5(out)
        

        out = self.up(out)
        # print(out.shape)
        out = torch.cat((out, t4), 1)
        out = self.decoder1(out)
        # output1 = self.map1(out)
        
        out = self.up(out)
        out = torch.cat((out, t3), 1)
        out = self.decoder2(out)
        # output2 = self.map2(out)
        
        out = self.up(out)
        out = torch.cat((out, t2), 1)
        out = self.decoder3(out)
        # output3 = self.map3(out)
        
        out = self.up(out)
        out = torch.cat((out, t1), 1)
        out = self.decoder4(out)
        # output4 = self.map4(out)
        
        out = self.segmentation_head(out)
        return out
        
