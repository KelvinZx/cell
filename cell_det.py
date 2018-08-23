import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from module import Identity_block, Strided_block, Res_Module, Conv_1x1, Dilated_Det_Module
from config import Config


class DetNetEncoder(nn.Module):
    def __init__(self):
        super(DetNetEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = Conv_1x1(64, 2)
        self.conv3 = Conv_1x1(128, 2)
        self.conv4 = Conv_1x1(128, 5)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.BatchNorm2d(2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(5, 5, 2, stride=2),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(5, 5, 2, stride=2),
            nn.BatchNorm2d(5)
        )
        self.Module1 = Res_Module(32, 32, downsample=False)
        self.Module2 = Res_Module(32, 64, downsample=True)
        self.Module3 = Res_Module(64, 128, downsample=True)
        self.Module4 = Res_Module(128, 256, downsample=True)
        self.Module5 = Dilated_Det_Module(256)
        self.Module6 = Dilated_Det_Module(256)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x_module1 = self.Module1(x)
        x_module2 = self.Module2(x_module1)
        x_module3 = self.Module3(x_module2)
        x_module4 = self.Module3(x_module3)
        x_module5 = self.Module3(x_module4)
        x_module6 = self.Module3(x_module5)
        return x_module1. x_module2, x_module3, x_module4, x_module5, x_module6


class DetNetDecoder(nn.Module):
    def __init__(self, channels=[64, 128, 256]):
        super(DetNetDecoder, self).__init__()
        m2_channel, m3_channel, m456_channel = channels[0], channels[1], channels[2]
        self.m2_conv1 = nn.Conv2d(m2_channel, m2_channel, kernel_size=1, padding=0)
        self.m3_conv1 = nn.Conv2d(m3_channel, m3_channel, kernel_size=1, padding=0)
        self.m4_conv1 = nn.Conv2d(m456_channel, m456_channel, kernel_size=1, padding=0)
        self.m5_conv1 = nn.Conv2d(m456_channel, m456_channel, kernel_size=1, padding=0)
        self.m6_conv1 = nn.Conv2d(m456_channel, m456_channel, kernel_size=1, padding=0)

    def forward(self, *input):
        m2_encoder, m3_encoder, \
        m4_encoder, m5_encoder, m6_encoder = input[0], input[1], \
                                             input[2], input[3], input[4]
        m2_encoder = self.m2_conv1(m2_encoder)
        m3_encoder = self.m3_conv1(m3_encoder)
        m4_encoder = self.m4_conv1(m4_encoder)
        m5_encoder = self.m5_conv1(m5_encoder)
        m6_encoder = self.m6_conv1(m6_encoder)


