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
        return x_module2, x_module3, x_module4, x_module5, x_module6


class DetNetDecoder(nn.Module):
    def __init__(self, channels=[64, 128, 256]):
        super(DetNetDecoder, self).__init__()
        m2_channel, m3_channel, m456_channel = channels[0], channels[1], channels[2]
        self.m2_conv1 = nn.Conv2d(m2_channel, m2_channel, kernel_size=1, padding=0)
        self.m2_bn1 = nn.BatchNorm2d(m2_channel)
        self.m3_conv1 = nn.Conv2d(m3_channel, m3_channel, kernel_size=1, padding=0)
        self.m3_bn1 = nn.BatchNorm2d(m3_channel)
        self.m4_conv1 = nn.Conv2d(m456_channel, m456_channel, kernel_size=1, padding=0)
        self.m4_bn1 = nn.BatchNorm2d(m456_channel)
        self.m5_conv1 = nn.Conv2d(m456_channel, m456_channel, kernel_size=1, padding=0)
        self.m5_bn1 = nn.BatchNorm2d(m456_channel)
        self.m6_conv1 = nn.Conv2d(m456_channel, m456_channel, kernel_size=1, padding=0)
        self.m6_bn1 = nn.BatchNorm2d(m456_channel)
        self.relu_456 = nn.ReLU(inplace=True)
        
        self.upsample_456 = nn.ConvTranspose2d(m456_channel, m3_channel, kernel_size=3, stride=2, padding=x)
        self.upsample_456_bn = nn.BatchNorm2d(m3_channel)
        self.relu_3456 = nn.ReLU(inplace=True)
        self.upsample_3456 = nn.ConvTranspose2d(m3_channel, m2_channel, kernel_size=3, stride=2, padding=x)
        self.upsample_3456_bn = nn.BatchNorm2d(m2_channel)
        self.relu_23456 = nn.ReLU(inplace=True)


    def forward(self, *input):
        m2_decoder, m3_decoder, \
        m4_decoder, m5_decoder, m6_decoder = input[0], input[1], \
                                             input[2], input[3], input[4]
        m2_decoder = self.m2_conv1(m2_decoder)
        m2_decoder = self.m2_bn1(m2_decoder)

        m3_decoder = self.m3_conv1(m3_decoder)
        m3_decoder = self.m3_bn1(m3_decoder)

        m4_decoder = self.m4_conv1(m4_decoder)
        m4_decoder = self.m4_bn1(m4_decoder)

        m5_decoder = self.m5_conv1(m5_decoder)
        m5_decoder = self.m5_bn1(m5_decoder)

        m6_decoder = self.m6_conv1(m6_decoder)
        m6_decoder = self.m6_bn1(m6_decoder)
        ######### merge decoder part from upper stage
        stage_56 = m5_decoder + m6_decoder
        stage_456 = stage_56 + m4_decoder
        stage_456 = self.relu_456(stage_456)
        
        stage_456_upsample = self.upsample_456(stage_456)
        stage_456_upsample_bn = self.upsample_456_bn(stage_456_upsample)
        stage_3456 = stage_456_upsample_bn + m3_decoder
        stage_3456 = self.relu_3456(stage_3456)
        stage_3456_upsample = self.upsample_3456(stage_3456)
        stage_3456_upsample_bn = self.upsample_3456_bn(stage_3456_upsample)

        stage_23456 = stage_3456_upsample_bn + m2_decoder
        stage_23456 = self.relu_23456(stage_23456)

        return m6_decoder, stage_56, stage_456, stage_3456, stage_23456


class DecoderRecovery(nn.Module):
    def __init__(self, channels=[64, 128, 256]):
        super(DecoderRecovery, self).__init__()
        m2_channel, m3_channel, m456_channel = channels[0], channels[1], channels[2]
        self.m6_deconv = nn.Sequential(
            nn.ConvTranspose2d(m456_channel, m456_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m456_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m456_channel, m456_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m456_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m456_channel, m456_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m456_channel)
        )
        self.m5_deconv = nn.Sequential(
            nn.ConvTranspose2d(m456_channel, m456_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m456_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m456_channel, m456_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m456_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m456_channel, m456_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m456_channel)
        )
        self.m4_deconv = nn.Sequential(
            nn.ConvTranspose2d(m456_channel, m456_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m456_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m456_channel, m456_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m456_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m456_channel, m456_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m456_channel)
        )
        self.m3_deconv = nn.Sequential(
            nn.ConvTranspose2d(m3_channel, m3_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m3_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m3_channel, m3_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m3_channel)
        )
        self.m2_deconv = nn.Sequential(
            nn.ConvTranspose2d(m2_channel, m2_channel, kernel_size=3, stride=2, padding=4),
            nn.BatchNorm2d(m2_channel)
        )
        self.channelsum = m2_channel+m3_channel+m456_channel*3
        self.concat_conv = nn.Conv2d(self.channelsum, 2, kernel_size=1, stride=1, padding=2)
        self.concat_bn = nn.BatchNorm2d(2)
        self.softmax = nn.Softmax(2)

    def forward(self, *input):
        m6_decoder, stage_56, stage_456, stage_3456, stage_23456 = input[0], input[1], input[2], input[3], input[4]
        stage_6 = self.m6_deconv(m6_decoder)
        stage_56 = self.m5_deconv(stage_56)
        stage_456 = self.m4_deconv(stage_456)
        stage_3456 = self.m3_deconv(stage_3456)
        stage_23456 = self.m2_deconv(stage_23456)
        x_concat = stage_23456+stage_3456+stage_456+stage_56+stage_6
        x_concat = self.concat_conv(x_concat)
        x_concat = self.concat_bn(x_concat)
        out = self.softmax(x_concat)
        return out


class CellDetection(nn.Module):
    def __init__(self):
        super(CellDetection, self).__init__()
        self.encoder = DetNetEncoder()
        self.decoder = DetNetDecoder()
        self.decoder_recover = DecoderRecovery()

    def forward(self, *input):
        x = input[0]
        out = self.decoder_recover(self.decoder_recover(self.encoder(x)))
        return out

    



