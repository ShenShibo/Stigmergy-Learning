# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import math
from torch._jit_internal import weak_module

# LeNet
class NaiveNet(nn.Module):
    def __init__(self, is_BN=False):
        super(NaiveNet, self).__init__()
        self.BN = is_BN
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        if self.BN:
            self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
        )
        if self.BN:
            self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        if self.BN:
            x = self.bn1(x)
        x = F.avg_pool2d(F.relu(x), kernel_size=2, stride=2)

        x = self.conv2(x)
        if self.BN:
            x = self.bn2(x)
        x = F.avg_pool2d(F.relu(x), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        out = self.fc3(self.fc2(self.fc1(x)))
        return F.softmax(out, dim=1)

    def test(self):
        pass

    def stigmergy(self):
        pass


# channel dropout
class DropoutNet(nn.Module):
    def __init__(self, p=0.5):
        super(DropoutNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=p, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(7 * 7 * 128, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def test(self):
        self.training = False
        self.dropout1.training = False

    def forward(self, x):
        self._stigmergy()
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=1)

    def _stigmergy(self):
        if self.training is False:
            return
        wg = self.conv2.weight.grad
        if wg is None:
            return
        else:
            temp_norm = torch.norm(wg.data, dim=(2, 3))
            influence_values = temp_norm.sum(dim=0)


# stigmergy network
class StigmergyNet(nn.Module):
    def __init__(self, p):
        super(StigmergyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.scale = 1.
        self.cMask = torch.Tensor(1, 128, 1, 1)
        self.cMask.fill_(1.)
        self.state_value = torch.ones(128)
        self.state_value = F.softmax(self.state_value, dim=0)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p = p
        self.training = True
        self.fc1 = nn.Linear(7 * 7 * 128, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        self._stigmergy()
        # self._dropout()
        x = x * self.cMask
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))

        return F.softmax(out, dim=1)

    def _dropout(self):
        assert self.p > 0. and self.p < 1.
        p = self.p
        dim = 128
        if self.training:
            self.cMask.fill_(0.)
            # 随机选择一半特征图置0
            index = torch.randperm(dim)[:int(dim * p)]
            self.cMask[:, index, :, :] = 1.
            # print(self.cMask.view(128))
        else:
            self.cMask.fill_(p)

    def _stigmergy(self):
        dim = 128
        eta = 0.9
        assert self.p > 0. and self.p < 1.
        p = self.p
        if self.training is False:
            return
        wg = self.conv2.weight.grad
        end = int(dim * p)
        if wg is None:
            self.cMask.fill_(0.)
            index = torch.randperm(dim)[:end]
            self.cMask[:, index, :, :] = 1.
            return
        else:
            # l2 norm表示改变策略
            temp_norm = torch.norm(wg.data, dim=(2, 3))
            # 对所有卷积核求和
            influence_values = temp_norm.sum(dim=0)
            # 参与计算的通道index
            temp_if = influence_values[influence_values > 0.]
            # 计算中值
            median = torch.median((temp_if).exp())
            # 减去中值(也可以减去均值)
            influence_values = self.scale * ((influence_values).exp() - median)
            # 乘以掩码，去掉未参与通道数
            influence_values = self.cMask.view(dim) * influence_values
            influence_values[influence_values > 0.] = 1./512
            influence_values[influence_values < 0.] = -1./512
            self.state_value = eta * self.state_value + influence_values
            # 升序排列，排列好的数值在原先tensor中的索引
            index = torch.argsort(self.state_value, descending=True)[:end]
            print(index)
            # 生成掩码
            self.cMask.fill_(0.)
            self.cMask[:, index, :, :] = 1.

    def cuda(self, device=None):
        self.state_value = self.state_value.cuda(device=device)
        self.cMask = self.cMask.cuda(device=device)
        return self._apply(lambda t: t.cuda(device))


class WCDNetwork(nn.Module):
    def __init__(self):
        super(WCDNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.scale = 1.
        self.cMask = torch.Tensor(1, 128, 1, 1)
        self.cMask.fill_(self.scale)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.training = True
        self.fc1 = nn.Linear(7 * 7 * 128, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        self.mask(x.clone())
        x = x * self.cMask
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))

        return F.softmax(out, dim=1)

    def mask(self, x):

        if self.training is False:
            self.cMask.fill_(1.)
            return
        b, c, _, _ = x.size()
        score = torch.mean(x, dim=(0, 2, 3))
        # 简单将一个batch的图按通道求和
        M = 64
        re_score = 1. / (score + 0.000000001)
        key = torch.pow(torch.rand(c).cuda(), re_score)
        index = torch.argsort(key, descending=True)[:M]
        alpha = score.sum() / score[index].sum()
        self.cMask.fill_(0.)
        self.cMask[:, index, :, :] = alpha.data

    def cuda(self, device=None):
        # self.state_value = self.state_value.cuda(device=device)
        self.cMask = self.cMask.cuda(device=device)
        return self._apply(lambda t: t.cuda(device))


class SEnet(nn.Module):
    def __init__(self):
        print("SEnet!")
        super(SEnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(128, 128 // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128 // 16, 128, bias=False),
            nn.Sigmoid()
        )
        # self.scale = 1.
        # self.cMask = torch.Tensor(1, 128, 1, 1)
        # self.cMask.fill_(self.scale)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.training = True
        self.fc1 = nn.Linear(7 * 7 * 128, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        # SEblock
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.SEblock(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))

        return F.softmax(out, dim=1)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        # input channels : 3
        # output channels : 64
        # channel size : 32x32
        self.stage1 = nn.Sequential(OrderedDict([
            ('conv1', conv3x3(3, 64)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('conv2', conv3x3(64, 64)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU())
        ]))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # input channels : 64
        # output channels : 128
        # channel size : 16x16
        self.stage2 = nn.Sequential(OrderedDict([
            ('conv3', conv3x3(64, 128)),
            ('bn3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU()),
            ('conv4', conv3x3(128, 128)),
            ('bn4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU())
        ]))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # input channels : 128
        # output channels : 256
        # channel size : 8x8
        self.stage3 = nn.Sequential(OrderedDict([
            ('conv5', conv3x3(128, 256)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU()),
            ('conv6', conv3x3(256, 256)),
            ('bn6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU()),
            ('conv7', conv3x3(256, 256)),
            ('bn7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU()),
        ]))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # input channels : 256
        # output channels : 512
        # channel size : 4x4
        self.stage4 = nn.Sequential(OrderedDict([
            ('conv8', conv3x3(256, 512)),
            ('bn8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU()),
            ('conv9', conv3x3(512, 512)),
            ('bn9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU()),
            ('conv10', conv3x3(512, 512)),
            ('bn10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU())
        ]))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # input channels : 512
        # output channels : 512
        # channel size : 2x2
        self.stage5 = nn.Sequential(OrderedDict([
            ('conv11', conv3x3(512, 512)),
            ('bn11', nn.BatchNorm2d(512)),
            ('relu11', nn.ReLU()),
            ('conv12', conv3x3(512, 512)),
            ('bn12', nn.BatchNorm2d(512)),
            ('relu12', nn.ReLU()),
            ('conv13', conv3x3(512, 512)),
            ('bn13', nn.BatchNorm2d(512)),
            ('relu13', nn.ReLU())
        ]))
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        # self.fc1 = nn.Linear(in_features=2 * 2 * 512, out_features=512)
        self.fc1 = nn.Linear(in_features=512, out_features=10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool1(x)

        x = self.stage2(x)
        x = self.pool2(x)

        x = self.stage3(x)
        x = self.pool3(x)

        x = self.stage4(x)
        x = self.pool4(x)

        x = self.stage5(x)
        x = self.pool5(x)
        x = x.view(x.size(0), -1)

        out = F.relu(self.fc1(x))
        return out


class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, p=0.5):
        super(MaskConv2d, self).__init__(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=bias)
        # filter mask
        self.fMask = torch.Tensor(in_channels, out_channels//groups, 1, 1)
        self.p = p

    def forward(self, input):
        if self.training is True:
            self.fMask = torch.rand(self.out_channels, self.in_channels, 1, 1)
            self.fMask[self.fMask > self.p] = 1.
            self.fMask[self.fMask <= self.p] = 0.
            return F.conv2d(input, self.weight * self.fMask, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        else:
            self.fMask.fill_(self.p)
            return F.conv2d(input, self.fMask * self.weight, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = MaskConv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = MaskConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.training = True
        self.fc1 = nn.Linear(7 * 7 * 128, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))

        return out

