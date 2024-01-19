import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import argparse
from datasets import *
from torch.optim.lr_scheduler import MultiStepLR
from skimage.measure.simple_metrics import compare_psnr
from SwinIR import *


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, mid_channel=None):
        super(DoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channel, out_channel, mid_channel=int(in_channel // 2))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(unet, self).__init__()
        self.inc = DoubleConv(in_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channel)

    def forward(self, x):
        input = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        res = self.outc(x)
        out = input - res
        return input, res, out


class unet_32(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(unet_32, self).__init__()
        self.inc = DoubleConv(in_channel, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)

        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, out_channel)

    def forward(self, x):
        input = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = input - out
        return x, x1, out

class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channel, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)

        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, out_channel)
        self.up11 = Up(512, 128)
        self.up22 = Up(256, 64)
        self.up33 = Up(128, 32)
        self.up44 = Up(64, 32)
        self.outc2 = OutConv(32, out_channel)


    def forward(self, x):
        input = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out1 = self.outc(x)

        x = self.up11(x5, x4)
        x = self.up22(x, x3)
        x = self.up33(x, x2)
        x = self.up44(x, x1)
        out2 = self.outc(x)
        img = input - out1 - out2
        return out1, out2, img


class Attfus(nn.Module):
    def __init__(self, in_channel, img_size, out_channel, rain=True):
        super(Attfus, self).__init__()
        self.inRain = rain
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv = DoubleConv(in_channel, out_channel, mid_channel=int(in_channel // 2))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0),
            nn.LeakyReLU(True)
        )
        self.img_size = img_size
        inter_channel = in_channel // 2
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=8, in_chans=0, embed_dim=in_channel)

        self.block = nn.ModuleList([
            SwinTransformerBlock(dim=in_channel, input_resolution=img_size,
                                 num_heads=4, window_size=4 if self.inRain else 8,
                                 shift_size=0 if self.inRain else 8 // 2)
            for i in range(1)
        ])

        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=8, in_chans=0, embed_dim=in_channel)
        if self.inRain:
            self.local_att = nn.Sequential(
                nn.Conv2d(in_channel, inter_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channel, in_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channel),
            )
        else:
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channel, inter_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channel, in_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channel)
            )
        self.down = DoubleConv(in_channel * 2, in_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3=None):
        if not x3 is None:
            x1 = self.down(torch.cat([x1, x3], dim=1))
        x1 = self.up(x1)
        x = x1 + x2
        _, _, H, W = x.shape

        if self.inRain:
            x = self.local_att(x)
        else:
            x = self.global_att(x)

        sig = self.sigmoid(x)

        x = x1 * sig + x2 * (1 - sig)

        x_ = self.patch_embed(x)
        for block in self.block:
            x_ = block(x_, (H, W))
        x_ = self.patch_unembed(x_, (H, W))
        x = self.conv(x + x_)
        return x


class ADUNet(nn.Module):

    def __init__(self, in_c, out_c):
        super(ADUNet, self).__init__()

        self.inc = DoubleConv(in_c, 32)
        self.down1 = Down(32, 64)  # 128 256
        self.down2 = Down(64, 128)  # 64 128
        self.down3 = Down(128, 256)  # 32 64
        self.down4 = Down(256, 256)  # 16 32
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upRain1 = Attfus(256, (32, 64), 128)
        self.upRain2 = Attfus(128, (64, 128), 64)
        self.upRain3 = Attfus(64, (128, 256), 32)
        self.upRain4 = Attfus(32, (256, 512), 16)
        self.outRain = nn.Conv2d(16, out_c, 1, 1, 0)

        self.upHaze1 = Attfus(256, (32, 64), 128, False)
        self.upHaze2 = Attfus(128, (64, 128), 64, False)
        self.upHaze3 = Attfus(64, (128, 256), 32, False)
        self.upHaze4 = Attfus(32, (256, 512), 16, False)
        self.outHaze = nn.Conv2d(16, out_c, 1, 1, 0)

    def forward(self, x):
        # x = (x - self.mean) / self.std
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        rain1 = self.upRain1(x5, x4)
        haze1 = self.upHaze1(x5, x4)
        rain2 = self.upRain2(rain1, x3, haze1)
        haze2 = self.upHaze2(haze1, x3, rain1)
        rain3 = self.upRain3(rain2, x2, haze2)
        haze3 = self.upHaze3(haze2, x2, rain2)
        rain4 = self.upRain4(rain3, x1, haze3)
        haze4 = self.upHaze4(haze3, x1, rain3)
        rain = self.outRain(rain4)
        haze = self.outHaze(haze4)

        out = x - rain - haze

        return rain, haze, out

class ADUNet_plus(nn.Module):

    def __init__(self, in_c, out_c):
        super(ADUNet_plus, self).__init__()


        self.inc = DoubleConv(in_c, 64)
        self.down1 = Down(64, 128)  # 128 256
        self.down2 = Down(128, 256)  # 64 128
        self.down3 = Down(256, 512)  # 32 64
        self.down4 = Down(512, 512)  # 16 32
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upRain1 = Attfus(512, (32, 64), 256)
        self.upRain2 = Attfus(256, (64, 128), 128)
        self.upRain3 = Attfus(128, (128, 256), 64)
        self.upRain4 = Attfus(64, (256, 512), 32)
        self.outRain = nn.Conv2d(32, out_c, 1, 1, 0)

        self.upHaze1 = Attfus(512, (32, 64), 256, False)
        self.upHaze2 = Attfus(256, (64, 128), 128, False)
        self.upHaze3 = Attfus(128, (128, 256), 64, False)
        self.upHaze4 = Attfus(64, (256, 512), 32, False)
        self.outHaze = nn.Conv2d(32, out_c, 1, 1, 0)

    def forward(self, x):
        # x = (x - self.mean) / self.std
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        rain1 = self.upRain1(x5, x4)
        haze1 = self.upHaze1(x5, x4)
        rain2 = self.upRain2(rain1, x3, haze1)
        haze2 = self.upHaze2(haze1, x3, rain1)
        rain3 = self.upRain3(rain2, x2, haze2)
        haze3 = self.upHaze3(haze2, x2, rain2)
        rain4 = self.upRain4(rain3, x1, haze3)
        haze4 = self.upHaze4(haze3, x1, rain3)
        rain = self.outRain(rain4)
        haze = self.outHaze(haze4)

        out = x - rain - haze

        return rain, haze, out

if __name__ == '__main__':
    from torchsummary import summary
    model = ADUNet(3, 3).cuda()
    from thop import profile

    dummy_input = torch.randn(1, 3, 256, 512).cuda()
    flops, parm = profile(model, (dummy_input,))
    print(f"flops: {flops} parm: {parm}")
    # model.eval()
    # summary(model, input_size=(3, 256, 512), batch_size=-1)
    # x = torch.randn((1, 3, 256, 512))
    # x = x.cuda()
    # _, _, y = model(x)
    # print(f"input:{x.shape} output:{y.shape}")
