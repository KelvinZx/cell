import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import module
from module import Identity_block, Strided_block, Conv_1x1
from config import Config

epsilon = 1e-7

class Res_Module(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(Res_Module, self).__init__()
        if downsample:
            self.conv1 = Strided_block(in_ch, out_ch)
        else:
            self.conv1 = Identity_block(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            Identity_block(out_ch, out_ch),
            Identity_block(out_ch, out_ch),
            Identity_block(out_ch, out_ch),
            Identity_block(out_ch, out_ch),
            Identity_block(out_ch, out_ch),
            Identity_block(out_ch, out_ch),
            Identity_block(out_ch, out_ch),
            Identity_block(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        self.Module4 = Res_Module(128, 128, downsample=False)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Module1(x)
        x = self.Module2(x)
        y1 = self.conv2(x)
        x = self.Module3(x)
        y2 = self.conv3(x)
        d1 = self.deconv1(y2)
        s1 = y1 + d1
        d2 = self.deconv2(s1)
        det = self.out(d2)

        x = self.Module4(x)
        y3 = self.conv4(x)
        d3 = self.deconv3(y3)
        d4 = self.deconv4(d3)
        cls = self.out(d4)
        return det, cls

class Attention_Net(nn.Module):
    def __init__(self):
        super(Attention_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = Conv_1x1(64, 32)
        self.conv3 = Conv_1x1(128, 32)
        self.conv4 = Conv_1x1(128, 32)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 2, 2, stride=2),
            nn.BatchNorm2d(2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 5, 2, stride=2),
            nn.BatchNorm2d(5)
        )
        self.Module1 = Res_Module(32, 32, downsample=False)
        self.Module2 = Res_Module(32, 64, downsample=True)
        self.Module3 = Res_Module(64, 128, downsample=True)
        self.Module4 = Res_Module(128, 128, downsample=False)
        self.att_conv = nn.Conv2d(32, 1, 1)
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Module1(x)
        x = self.Module2(x)
        y1 = self.conv2(x)
        x = self.Module3(x)
        y2 = self.conv3(x)
        d1 = self.deconv1(y2)
        s1 = y1 + d1
        att1 = self.att_conv(s1)
        att1 = att1.view(att1.size(0), -1)
        att1 = self.softmax(att1)
        att1 = att1.view(att1.size(0), 1 , 128, 128)
        x = self.Module4(x)
        y3 = self.conv4(x)
        d3 = self.deconv3(y3)
        att2 = self.att_conv(d3)
        att2 = att2.view(att2.size(0), -1)
        att2 = self.softmax(att2)
        att2 = att1.view(att2.size(0), 1, 128, 128)
        d3_att = d3 * att1

        s1_att = s1 * att2
        d2 = self.deconv2(s1_att)
        det = self.out(d2)
        d4 = self.deconv4(d3_att)
        cls = self.out(d4)
        return det, cls

class Attention_Net_Global(nn.Module):
    def __init__(self):
        super(Attention_Net_Global, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = Conv_1x1(64, 32)
        self.conv3 = Conv_1x1(128, 32)
        self.conv4 = Conv_1x1(128, 32)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 2, 2, stride=2),
            nn.BatchNorm2d(2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 5, 2, stride=2),
            nn.BatchNorm2d(5)
        )
        self.Module1 = Res_Module(32, 32, downsample=True)
        self.Module2 = Res_Module(32, 64, downsample=True)
        self.Module3 = Res_Module(64, 128, downsample=True)
        self.Module4 = Res_Module(128, 128, downsample=False)

        self.att_conv = nn.Conv2d(32, 1, 1)
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.LogSoftmax(dim=1)
        self.att_global = module.AttentionGlobal(inchannel=32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Module1(x)
        x = self.Module2(x)
        y1 = self.conv2(x)
        x = self.Module3(x)
        y_m4 = self.Module4(x)
        y2 = self.conv3(y_m4)
        d1 = self.deconv1(y2)

        s1 = y1 + d1
        att1 = self.att_global(y2)
        x = self.Module4(x)
        y3 = self.conv4(x)
        cls_att = att1 * y3

        det_2 = self.deconv3(s1)
        det_1 = self.deconv2(det_2)
        det  =self.out(det_1)

        cls_4 = self.deconv3(cls_att)
        cls_2 = self.deconv3(cls_4)
        cls_1 = self.deconv4(cls_2)
        cls = self.out(cls_1)
        '''
        d3_att = att1
        #s1_att = s1 * att2
        s1_att = att2
        d2_1 = self.deconv3(s1_att)
        d2 = self.deconv2(d2_1)
        det = self.out(d2)
        d4_1 = self.deconv3(d3_att)
        d4 = self.deconv4(d4_1)
        cls = self.out(d4)
        '''
        return det, cls
