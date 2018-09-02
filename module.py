import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time


def _isArrayLike(obj):
    """
    check if this is array like object.
    """
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Renet(nn.Module):
    """
    This Renet is implemented according to paper
    """
    def __init__(self, inchannel, LSTM_channel = 32, outchannel = 256):
        """
        patch size = 1 and LSTM channel = 256 is default setting according to the origin code.
        :param inplane: input channel size.
        :param outchannel: output channel size
        :param patch_size: num of patch to be cut.
        :param LSTM_channel: filters for LSTM.
        """

        #################
        # Warning! The outchannel should be equal to Width * Height of current feature map,
        # Please change this number manually.
        #################
        super(Renet, self).__init__()
        self.vertical_LSTM = nn.LSTM(input_size=inchannel,
                                     hidden_size=LSTM_channel,
                                     batch_first=True,
                                     bidirectional=True)
        self.horizontal_LSTM = nn.LSTM(input_size=2 * LSTM_channel,
                                       hidden_size=LSTM_channel,
                                       batch_first=True,
                                       bidirectional=True)
        self.conv = nn.Conv2d(2 * LSTM_channel, outchannel, 1)
        self.bn = nn.BatchNorm2d(outchannel)

    def forward(self, *input):
        x = input[0]
        vertical_concat = []
        size = x.size()
        width, height = size[2], size[3]
        assert width == height
        x = torch.transpose(x, 1, 3)
        for i in range(width):
            h, _ = self.vertical_LSTM(x[:, :, i, :])
            vertical_concat.append(h)
        x = torch.stack(vertical_concat, dim=2)
        #print(x.size())
        horizontal_concat = []
        for i in range(width):
            h, _ = self.horizontal_LSTM(x[:, i, :, :])
            horizontal_concat.append(h)
        x = torch.stack(horizontal_concat, dim=3)
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        out = self.bn(x)
        #print(out.size())
        return out


class AttentionGlobal(nn.Module):
    """
    Global Attention module.
    """
    def __init__(self,
                 inchannel,
                 att_dilation = 3,
                 renet_LSTM_channel=256,
                 global_feature = 10):
        super(AttentionGlobal, self).__init__()
        # outplane should be height * width
        self.inchannel = inchannel
        self.global_feature = global_feature
        self.att_dilation = att_dilation
        self.renet = Renet(inchannel=inchannel, LSTM_channel=renet_LSTM_channel,
                           outchannel=global_feature**2) # Set the LSTM channel and output channel.

    def forward(self, *input):
        x = input[0]
        #print(x.size())
        size = x.size()
        assert self.inchannel == size[1]
        kernel = self.renet(x)
        kernel = F.softmax(kernel, 1) #do softmax along channel axis.
        kernel = kernel.reshape(size[0] * size[2] * size[3], 1, 1, self.global_feature, self.global_feature)
        #print('kernel:', kernel.size())
        x = torch.unsqueeze(x, 0)
        #print(x.size())
        x = F.conv3d(input=x, weight=kernel, bias=None, stride=1,
                     padding=0, dilation=(1, self.att_dilation, self.att_dilation), groups=size[0])
        #print(x.size())

        out = torch.reshape(x, (size[0], self.inchannel, size[2], size[3]))
        return out


class _AttentionLocal_Conv(nn.Module):
    def __init__(self, inchannel, conv1_filter, conv1_dilation, conv1_outchannel,
                 conv_last_outchannel=49,
                 num_conv=2):
        """

        :param inchannel: input channel.
        :param conv1_filter: conv1 filter, current is 7
        :param conv1_dilation: conv1 dilation, current is 2
        :param conv1_outchannel: conv1 output channel, current is 128
        :param conv2_filter:  conv2 output filter, current is 1
        :param conv2_dilation: conv2 dilation, current 1.
        :param conv2_outchannel: conv2 output channel, current 49
        :param num_conv: Number of convolution layer before softmax, current number according to paper is 2.
        """
        super(_AttentionLocal_Conv, self).__init__()
        self.num_conv = num_conv
        self.padding1 = (conv1_dilation * (conv1_filter-1) - 1) /2
        self.padding2 = (1 * (1 - 1) - 1) / 2
        self.conv1 = nn.Conv2d(inchannel, conv1_outchannel, conv1_filter,
                               dilation=conv1_dilation, padding=self.padding1)
        self.bn1 = nn.BatchNorm2d(conv1_outchannel)
        self.conv_last = nn.Conv2d(conv1_outchannel, conv_last_outchannel, 1,
                               dilation=1, padding=0)
        self.bn_last = nn.BatchNorm2d(conv_last_outchannel)

    def forward(self, *input):
        x = input[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        return x


class AttentionLocal(nn.Module):
    """
    Local Attention module.
    """
    def __init__(self, inchannel, local_feature):
        """

        :param starting_point: left upper corner point
        :param width: Width of wanted feature map
        :param height: Height of wanted feature map
        :param kernels:
        :param dilation:
        """
        super(AttentionLocal, self).__init__()
        self.local_feature = local_feature
        self.inchannel = inchannel
        self._conv = _AttentionLocal_Conv(inchannel, conv1_filter=7, conv1_dilation=2,
                                          conv1_outchannel=256,
                                          conv_last_outchannel=local_feature**2)


    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self._conv(x)
        kernel = F.softmax(kernel, dim=1)
        kernel = kernel.reshape(size[0] * size[2] * size[3], 1, 1, self.local_feature, self.local_feature)
        x = torch.unsqueeze(x, 0)
        x = F.conv3d(input=x, weight=kernel, bias=None, stride=1,
                     padding=0, dilation=(1, self.att_dilation, self.att_dilation), groups=size[0])
        out = torch.reshape(x, (size[0], self.inchannel, size[2], size[3]))
        return out


class Identity_block(nn.Module):
    '''(Conv=>BN=>ReLU)*2'''

    def __init__(self, in_ch, out_ch):
        super(Identity_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.conv(x)
        y = residual + y
        y = self.relu(y)
        return y


class Strided_block(nn.Module):
    '''downsample featuremap between modules'''

    def __init__(self, in_ch, out_ch):
        super(Strided_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3 ,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.downsample(x)
        y = self.conv(x)
        y = residual + y
        y = self.relu(y)
        return y


class Conv_1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Res_Module(nn.Module):
    def __init__(self, in_ch, out_ch, block_num=9, downsample=True):
        super(Res_Module, self).__init__()
        self.block_num = block_num - 1
        if downsample:
            self.conv1 = Strided_block(in_ch, out_ch)
        else:
            self.conv1 = Identity_block(in_ch, out_ch)
        self.id_block = nn.Sequential(
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
        x = self.id_block(x)
        return x


class Dilated_bottleneck(nn.Module):
    """
    Dilated block without 1x1 convolution projection, structure like res-id-block
    """
    def __init__(self, channel, dilate_rate=2):
        super(Dilated_bottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, dilation=dilate_rate, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, *input):
        x = input[0]
        x_ori = x
        x = self.conv(x)
        x = x + x_ori
        x = self.relu(x)
        return x


class Dilated_with_projection(nn.Module):
    """
    Dilated block with 1x1 convolution projection for the shortcut, structure like res-conv-block
    """
    def __init__(self, channel, dilate_rate=2):
        super(Dilated_with_projection, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, dilation=dilate_rate, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0),
            nn.BatchNorm2d(channel),
        )
        self.relu = nn.ReLU()

    def forward(self, *input):
        x = input[0]
        x_ori = x
        x_ori = self.shortcut(x_ori)
        x = self.conv(x)
        x = x + x_ori
        x = self.relu(x)
        return x


class Dilated_Det_Module(nn.Module):
    """

    """
    def __init__(self, channel):
        super(Dilated_Det_Module, self).__init__()
        self.dilated_with_project = Dilated_with_projection(channel)
        self.dilate_bottleneck1 = Dilated_bottleneck(channel)
        self.dilate_bottleneck2 = Dilated_bottleneck(channel)

    def forward(self, *input):
        x = input[0]
        x = self.dilated_with_project(x)
        x = self.dilate_bottleneck1(x)
        x = self.dilate_bottleneck2(x)
        return x
