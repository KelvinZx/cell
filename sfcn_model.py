import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import os
import dataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import scipy.misc as misc
import cv2
import scipy.io as sio
import module

epsilon = 1e-7

class identity_block(nn.Module):
    '''(Conv=>BN=>ReLU)*2'''

    def __init__(self, in_ch, out_ch):
        super(identity_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        y = self.conv(x)
        y = residual + y
        y = self.relu(y)
        return y

class strided_block(nn.Module):
    '''downsample featuremap between modules'''

    def __init__(self, in_ch, out_ch):
        super(strided_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3 ,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        y = self.conv(x)
        y = residual + y
        y = self.relu(y)
        return y

class conv_1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Res_Module(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(Res_Module, self).__init__()
        if downsample:
            self.conv1 = strided_block(in_ch, out_ch)
        else:
            self.conv1 = identity_block(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
            identity_block(out_ch, out_ch),
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
        self.conv2 = conv_1x1(64, 2)
        self.conv3 = conv_1x1(128, 2)
        self.conv4 = conv_1x1(128, 5)
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
        self.conv2 = conv_1x1(64, 32)
        self.conv3 = conv_1x1(128, 32)
        self.conv4 = conv_1x1(128, 32)
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
        self.conv2 = conv_1x1(64, 32)
        self.conv3 = conv_1x1(128, 32)
        self.conv4 = conv_1x1(128, 32)
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
        self.att_global = module.AttentionGlobal(patch_size=16,inplane=32,outplane=1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Module1(x)
        x = self.Module2(x)
        y1 = self.conv2(x)
        x = self.Module3(x)
        y2 = self.conv3(x)
        d1 = self.deconv1(y2)
        s1 = y1 + d1
        att1 = self.att_global(s1)
        x = self.Module4(x)
        y3 = self.conv4(x)
        d3 = self.deconv3(y3)
        att2 = self.att_global(d3)
        d3_att = d3 * att1

        s1_att = s1 * att2
        d2 = self.deconv2(s1_att)
        det = self.out(d2)
        d4 = self.deconv4(d3_att)
        cls = self.out(d4)
        return det, cls

def train(model, weight_det=None, weight_cls=None,data_dir='', preprocess=True, gpu=True, batch_size=2, num_epochs=100, target_size=256):
    if weight_det == None:
        weight_det = torch.Tensor([1, 1])
    else:
        weight_det = torch.Tensor(weight_det)

    if weight_cls == None:
        weight_cls = torch.Tensor([1, 1, 1, 1, 1])
    else:
        weight_cls = torch.Tensor(weight_cls)

    writer = SummaryWriter()

    data = dataset.CRC_joint(data_dir, target_size=target_size)
    x_train, y_train_det, y_train_cls = data.load_train(preprocess=preprocess)
    train_count = len(x_train)

    x_val, y_val_det, y_val_cls = data.load_val(preprocess=preprocess)
    val_count = len(x_val)
    val_steps = int(val_count / batch_size)
    print('training imgs:', train_count)
    print('val imgs:', val_count)

    trainset = np.concatenate([x_train, y_train_det, y_train_cls], axis=1)
    trainset = torch.Tensor(trainset)

    valset = np.concatenate([x_val, y_val_det, y_val_cls], axis=1)
    valset = torch.Tensor(valset)

    if gpu:
        model = model.cuda()
        trainset = trainset.cuda()
        valset = valset.cuda()
        weight_det = weight_det.cuda()
        weight_cls = weight_cls.cuda()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    NLLLoss_det = nn.NLLLoss(weight=weight_det)
    NLLLoss_cls = nn.NLLLoss(weight=weight_cls)
    best_loss = 99999.0

    for epoch in range(num_epochs):

        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        for i, datapack in enumerate(train_loader, 0):
            train_imgs = datapack[:, 0:3]
            train_det_masks = datapack[:, 3:4]
            train_cls_masks = datapack[:, 4:]

            train_det_masks = train_det_masks.long()
            train_cls_masks = train_cls_masks.long()

            train_det_masks = train_det_masks.view(
                train_det_masks.size()[0],
                train_det_masks.size()[2],
                train_det_masks.size()[3]
            )

            train_cls_masks = train_cls_masks.view(
                train_cls_masks.size()[0],
                train_cls_masks.size()[2],
                train_cls_masks.size()[3]
            )

            optimizer.zero_grad()
            train_det_out, train_cls_out = model(train_imgs)
            t_det_loss = NLLLoss_det(train_det_out, train_det_masks)
            t_cls_loss = NLLLoss_cls(train_cls_out, train_cls_masks)
            t_loss = t_det_loss + t_cls_loss
            t_loss.backward()
            optimizer.step()
            train_loss += t_loss.item()

            if i % 10 == 9:
                print('epoch: %3d, step: %3d loss: %.5f' % (epoch + 1, i + 1, train_loss / 10))
                writer.add_scalar('train_loss', train_loss, (210 * epoch + i + 1))
                train_loss = 0.0

        for i, datapack in enumerate(val_loader, 0):
            val_imgs = datapack[:, 0:3]
            val_det_masks = datapack[:, 3:4]
            val_cls_masks = datapack[:, 4:]

            val_det_masks = val_det_masks.long()
            val_det_masks = val_det_masks.view(
                val_det_masks.size()[0],
                val_det_masks.size()[2],
                val_det_masks.size()[3]
            )

            val_cls_masks = val_cls_masks.long()
            val_cls_masks = val_cls_masks.view(
                val_cls_masks.size()[0],
                val_cls_masks.size()[2],
                val_cls_masks.size()[3]
            )

            # optimizer.zero_grad()
            val_det_out, val_cls_out = model(val_imgs)
            v_det_loss = NLLLoss_det(val_det_out, val_det_masks)
            v_cls_loss = NLLLoss_cls(val_cls_out, val_cls_masks)
            v_loss = v_det_loss + v_cls_loss
            val_loss += v_loss.item()

            if i % val_steps == val_steps - 1:
                val_loss = val_loss / val_steps
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), './ckpt/test_att_global.pkl')
                end = time.time()
                time_spent = end - start
                writer.add_scalar('val_loss', val_loss, epoch)
                print('epoch: %3d, time: %.5f val_loss: %.5f' % (epoch + 1, time_spent, val_loss))
                val_loss = 0.0
                #p, r, f = MyMetrics(model)
                #writer.add_scalar('precision', p, epoch)
                #writer.add_scalar('recall', r, epoch)
                #writer.add_scalar('f1_score', f, epoch)
                #print('p:', p)
                #print('r:', r)
                #print('f:', f)
                print('******************************************************************************')

    # writer.export_scalars_to_json('./loss.json')
    writer.close()
if __name__ == '__main__':
    net = Attention_Net_Global()
    train(net, weight_det=[0.1, 2], weight_cls=[0.1, 4, 3, 6, 10], data_dir='./aug', target_size=64)