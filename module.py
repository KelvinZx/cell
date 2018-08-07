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
    def __init__(self, inplane, outchannel = 256, patch_size = 1, LSTM_channel = 32):
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
        self.patch_size = patch_size
        self.horizontal_LSTM = nn.LSTM(input_size=inplane,
                                       hidden_size=LSTM_channel,
                                       batch_first=True,
                                       bidirectional=True)
        self.vertical_LSTM = nn.LSTM(input_size=inplane,
                                     hidden_size=LSTM_channel,
                                     batch_first=True,
                                     bidirectional=True)
        self.conv = nn.Conv2d(LSTM_channel, outchannel, 1)
        self.bn = nn.BatchNorm2d(outchannel)


    def forward(self, *input):
        # input is (batch, channel, width, height)

        # Here we follow PiCANet which we first flip horizontally twice,
        # Then vertically twice,
        x = input[0]
        print('x:',x.size())
        vertical_fwd_concat = []
        vertical_inv_concat = []
        horizon_fwd_concat = []
        horizon_inv_concat = []

        width, height = x.size()[2], x.size()[3]
        print('width:',width)
        width_per_patch = int(width / self.patch_size)
        print('width per patch:',width_per_patch)
        height_per_patch = int(height / self.patch_size)
        #assert width_per_patch.is_interger()
        #assert height_per_patch.is_interger()
        #######
        # use LSTM horizontally forward and backward
        for i in range(width):
            lstm_x = x[:, :, i]
            print('lstm_x:',lstm_x.size())
            lstm_input = lstm_x.view(lstm_x.size()[0], lstm_x.size()[1], -1)
            print('lstm_input:',lstm_input.size())
            horizon_fwd, _ = self.horizontal_LSTM(lstm_input)
            print('horizon_fwd:',horizon_fwd.size())
            horizon_fwd = horizon_fwd.view(horizon_fwd.size()[0],horizon_fwd.size()[1],horizon_fwd.size()[2],1)
            if i == 0:
                a = horizon_fwd
            else:
                a = torch.cat([a,horizon_fwd],dim=-1)
            #horizon_fwd_concat.append(horizon_fwd)
        #x_horizon_fwd = torch.stack(horizon_fwd_concat, dim=-1)
        print('x_horizon_fwd:',a.size())

        x = x_horizon_fwd
        #######
        # use LSTM vertically upward and downward
        for j in range(height):
            vertical_fwd, _ = self.vertical_LSTM(x_horizon_fwd[:, :, :,j])
            vertical_fwd_concat.append(vertical_fwd)
        print(len(vertical_fwd_concat))
        x_vertical_fwd = torch.stack(vertical_fwd_concat, dim=3)


        out = self.conv(x_vertical_fwd)
        out = self.bn(out)
        return out


class AttentionGlobal(nn.Module):
    """
    Global Attention module.
    """
    def __init__(self, patch_size, inplane, outplane, renet_outplane=100, renet_LSTM_channel=256):
        super(AttentionGlobal, self).__init__()
        # outplane should be height * width
        self.patch_size = patch_size
        self.renet = Renet(patch_size=patch_size, inplane=inplane,outchannel=outplane) # Set the LSTM channel and output channel.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, *input):
        x = input[0]
        #print(x.size())
        x_size = x.size()
        x_renet = self.renet(x)
        # reshape tensor of (batch, channel, w, h) to
        # (batch, 1, w, h) then do softmax
        x_renet = x_renet.view(1, 1)
        print(x_renet.size())
        x_renet = self.softmax(x_renet)
        out = x_renet + x
        return out


class AttentionLocal(nn.Module):
    """
    Local Attention module.
    """
    def __init__(self, starting_point, width, height, dilation=None):
        """

        :param starting_point: left upper corner point
        :param width: Width of wanted feature map
        :param height: Height of wanted feature map
        :param kernels:
        :param dilation:
        """
        super(AttentionLocal, self).__init__()
        self.sofxmax = F.softmax(input, dim=1)
        self.starting_point = starting_point
        self.width = width
        self.height = height

    def forward(self, *input):
        x_ori = input[0]
        x_att = input[1]
        x_att = x_att[:, self.starting_point: self.starting_point + self.width,
                        self.starting_point + self.height, :]
        #print(x_att.size())
        x_att = self.softmax(x_att)
        out = x_att + x_ori
        return out