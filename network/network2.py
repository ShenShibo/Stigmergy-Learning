# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import math

import time
import torchvision.models.resnet
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }


class Stack(object):
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def clear(self):
        del self.items[:]

    def empty(self):
        return self.size() == 0

    def size(self):
        return len(self.items)

    def top(self):
        return self.items[self.size()-1]


class Svgg(nn.Module):
    _default = [.5, .0, .0, .0, .0, .0, .0, .5, .5, .5, .5, .5, .0]
    def __init__(self,
                 num_classes=10,
                 update_round=1,
                 is_stigmergy=True,
                 ksai=0.9,
                 dr=None):
        super(Svgg, self).__init__()
        self.distance_matrices = []
        self.sv = []
        self.mask = []
        self.counter = []
        self.activation_stack = Stack()
        self.layer_index_stack = Stack()
        self.update_round = update_round
        self.stigmergy = is_stigmergy
        self.ksai = ksai
        if dr is None:
            self._dr = self._default
        else:
            self._dr = dr
        self.feature = self._make_layers(cfg['D'], bn=True)
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layers(self, cfg=[], bn=True):
        layers = []
        in_channels = 3
        # count = 0
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                self.distance_matrices.append(torch.eye(v))
                self.counter.append(torch.zeros(v, v))
                self.sv.append(torch.zeros(v))
                self.mask.append(torch.zeros(1, v, 1, 1))
                # conv2d = DropConv2d(in_channels, v, kernel_size=3, padding=1, dr=self._dr[count])
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # count += 1
                if bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, iterations):
        count = 0
        count2 = 0
        x.requires_grad_()
        for _, (_, m) in enumerate(self.feature._modules.items()):
            x = m(x)
            # save activations to compute channel utilities
            if self.training is True:
                if isinstance(m, nn.ReLU):
                    x.register_hook(self.compute_rank)
                    self.activation_stack.push(x)
                    self.layer_index_stack.push(count)
                    count += 1
            if isinstance(m, nn.Conv2d):
                end = int(m.out_channels * (1 - self._dr[count2]))
                self.mask[count2].fill_(0.)
                if self.stigmergy is False and self.training is True:
                    index = torch.randperm(m.out_channels)[:end]
                    self.mask[count2][:, index, :, :] = 1.
                else:
                    index = torch.argsort(self.sv[count2], descending=True)[:end]
                    self.mask[count2][:, index, :, :] = 1
                x = x * self.mask[count2].expand_as(x)
                count2 += 1

        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def compute_rank(self, grad):
        k = self.layer_index_stack.pop()
        activation = self.activation_stack.pop()
        b, c, h, w = activation.size()
        mask = self.mask[k].squeeze()
        # 求更新量
        temp = torch.abs(torch.sum(grad.data * activation.data, dim=(2, 3)))
        values = torch.sum(temp, dim=0) / (b * h * w)
        # 对更新量进行量l2正则化
        values /= torch.norm(values)
        epsilon = 0.000000001
        # 共识主动性
        temp1 = values.expand(c, c)
        temp2 = values.unsqueeze(dim=1).expand(c, c)
        temp = temp1 * temp2
        self.counter[k][temp > 0] += 1.
        temp3 = (self.counter[k] - 1) / (self.counter[k] + epsilon)
        temp3[temp==0.] = 1.
        # 更新距离矩阵
        self.distance_matrices[k] *= temp3
        self.distance_matrices[k] += (1-temp3) * temp
        self.distance_matrices[k][torch.eye(c) > 0.] = 1.
        # 共识主动更新量
        values = self.distance_matrices[k].mm(values.unsqueeze(dim=1))
        # 状态值更新
        self.sv[k][mask > 0.] *= self.ksai
        self.sv[k] += (1-self.ksai) * values.squeeze() * mask

    def cuda(self, device=None):
        DEVICE = torch.device('cuda:{}'.format(device))
        for i in range(len(self.sv)):
            self.distance_matrices[i] = self.distance_matrices[i].to(DEVICE)
            self.sv[i] = self.sv[i].to(DEVICE)
            self.mask[i] = self.mask[i].to(DEVICE)
            self.counter[i] = self.counter[i].to(DEVICE)
        return self._apply(lambda t: t.cuda(device))



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward1(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

    def forward2(self, x):
        out = self.conv2(x)
        out = self.bn2(out)
        return out

    def add_residual(self, x, y):
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        return self.relu(y)

    def forward(self, x):
        residual = x
        out = self.forward1(x)
        out = self.forward2(out)
        return self.add_residual(residual, out)


class SResNet(nn.Module):
    # 56 layers resnet
    _layer = [0.3, 0.3, 0.3]

    def __init__(self, num_classes=10, update_round=1, is_stigmergy=True, ksai=0.8):
        super(SResNet, self).__init__()
        self.distance_matrices = []
        self.sv = []
        self.mask = []
        self.counter = []
        self.activation_stack = Stack()
        self.layer_index_stack = Stack()
        self.update_round = update_round
        self.stigmergy = is_stigmergy
        self.ksai = ksai
        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 16, 9)
        self.layer2 = self._make_layer(BasicBlock, 32, 9, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 9, stride=2)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_para(self, rounds=1, planes=64):
        for i in range(rounds):
            self.distance_matrices.append(torch.eye(planes))
            self.counter.append(torch.zeros(planes, planes))
            self.sv.append(torch.zeros(planes))
            self.mask.append(torch.zeros(1, planes, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self._make_para(rounds=1, planes=planes)
        self.inplanes = planes
        self._make_para(rounds=1, planes=planes)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            self._make_para(rounds=2, planes=planes)

        return nn.Sequential(*layers)

    def forward(self, x, iterations):
        count = 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for l, layer in enumerate([self.layer1, self.layer2, self.layer3]):
            for _, (_, m) in enumerate(layer._modules.items()):
                residual = x
                x = m.forward1(x)
                if self.training is True:
                    x.register_hook(self.compute_rank)
                    self.activation_stack.push(x)
                    self.layer_index_stack.push(count)
                # forward channel selection
                end = int(m.conv1.out_channels * (1 - self._layer[l]))
                # mask construction
                self.mask[count].fill_(0.)
                if self.stigmergy is False and self.training is True:
                    index = torch.randperm(m.conv1.out_channels)[:end]
                    self.mask[count][:, index, :, :] = 1.
                else:
                    index = torch.argsort(self.sv[count], descending=True)[:end]
                    self.mask[count][:, index, :, :] = 1
                x = x * self.mask[count].expand_as(x)
                count += 1
                # inner layer
                x = m.forward2(x)
                if self.training is True:
                    x.register_hook(self.compute_rank)
                    self.activation_stack.push(x)
                    self.layer_index_stack.push(count)
                # forward channel selection
                end = int(m.conv2.out_channels * (1 - self._layer[l]))
                self.mask[count].fill_(0.)
                if self.stigmergy is False and self.training is True:
                    index = torch.randperm(m.conv2.out_channels)[:end]
                    self.mask[count][:, index, :, :] = 1.
                else:
                    index = torch.argsort(self.sv[count], descending=True)[:end]
                    self.mask[count][:, index, :, :] = 1
                x = x * self.mask[count].expand_as(x)
                residual = residual * self.self.mask[count].expand_as(x)
                x = m.add_residual(residual, x)
                count += 1
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def compute_rank(self, grad):
        k = self.layer_index_stack.pop()
        activation = self.activation_stack.pop()
        b, c, h, w = activation.size()

        mask = self.mask[k].squeeze()
        # 求更新量
        temp = torch.abs(torch.sum(grad.data * activation.data, dim=(2, 3)))
        values = torch.sum(temp, dim=0) / (b * h * w)
        # 对更新量进行量l2正则化
        values /= torch.norm(values)
        epsilon = 0.000000001
        # 共识主动性
        temp1 = values.expand(c, c)
        temp2 = values.unsqueeze(dim=1).expand(c, c)
        temp = temp1 * temp2
        self.counter[k][temp > 0] += 1.
        temp3 = (self.counter[k] - 1) / (self.counter[k] + epsilon)
        temp3[temp==0.] = 1.
        # 更新距离矩阵
        self.distance_matrices[k] *= temp3
        self.distance_matrices[k] += (1-temp3) * temp
        self.distance_matrices[k][torch.eye(c) > 0.] = 1.
        # 共识主动更新量
        values = self.distance_matrices[k].mm(values.unsqueeze(dim=1))
        # 状态值更新
        self.sv[k][mask > 0.] *= self.ksai
        self.sv[k] += (1-self.ksai) * values.squeeze() * mask

    def cuda(self, device=None):
        DEVICE = torch.device('cuda:{}'.format(device))
        for i in range(len(self.sv)):
            self.distance_matrices[i] = self.distance_matrices[i].to(DEVICE)
            self.sv[i] = self.sv[i].to(DEVICE)
            self.mask[i] = self.mask[i].to(DEVICE)
            self.counter[i] = self.counter[i].to(DEVICE)
        return self._apply(lambda t: t.cuda(device))



