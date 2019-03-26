# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import math
from torch._jit_internal import weak_module
from torch.nn.parameter import Parameter
import time

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

class Stack():
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
    _dr = [.1, .1, .2, .2, .3, .3, .3, .4, .4, .4, .5, .5, .5]
    def __init__(self,
                 num_classes=10,
                 update_round = 1,
                 is_stigmergy=True,
                 ksai = 0.9):
        super(Svgg, self).__init__()
        self.distance_matrices = []
        self.sv = []
        self.activation_stack = Stack()
        self.mask_stack = Stack()
        self.layer_index_stack = Stack()
        self.update_round = update_round
        self.stigmergy = is_stigmergy
        self.ksai = ksai
        self.rounds = 0
        self.feature = self._make_layers(cfg['D'], bn=True)
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layers(self, cfg=[], bn=True):
        layers = []
        in_channels = 3
        count = 0
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                self.distance_matrices.append(torch.zeros(in_channels, in_channels))
                self.sv.append(torch.zeros(in_channels))
                conv2d = DropConv2d(in_channels, v, kernel_size=3, padding=1, dr=self._dr[count])
                count += 1
                if bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, DropConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, iterations):
        count = 0
        flag = False
        # x = self.feature(x)
        x.requires_grad_()
        for _, (_, m) in enumerate(self.feature._modules.items()):
            if iterations % self.update_round == 0 and self.training is True:
                flag = True
                if isinstance(m, DropConv2d):
                    x.register_hook(self.compute_rank)
                    self.mask_stack.push(m.mask)
                    self.activation_stack.push(x)
                    self.layer_index_stack.push(count)
                    count += 1
            x = m(x)
        x = x.view(x.size(0), -1)
        if flag is True:
            self.rounds += 1
        return self.classifier(x)

    def compute_rank(self, grad):
        activation = self.activation_stack.pop()
        b, c, h, w = activation.size()
        mask = self.mask_stack.pop()
        mask = mask.view(mask.size(1))
        k = self.layer_index_stack.pop()
        # 求更新量
        temp = torch.abs(torch.sum(grad.data * activation.data, dim=(2, 3)))
        values = torch.sum(temp, dim=0) / (b * h * w)
        # 对更新量进行量l2正则化
        values /= torch.norm(values)
        num = (1-self._dr[k]) ** 2 * self.rounds
        if self.stigmergy:
            index = (mask > 0).nonzero()
            for i in index:
                for j in index:
                    if i == j:
                        continue
                    else:
                        self.distance_matrices[k][i, j] = (num / (num + 1)) *\
                                                          self.distance_matrices[k][i, j] +\
                                                          (1 / (num + 1)) *\
                                                          values[i] * values[j]
            values = (-self.distance_matrices[k]).exp().mm(values.unsqueeze(dim=1))
        # 状态值更新
        self.sv[k][mask > 0.] *= self.ksai
        self.sv[k] = self.sv[k]+ (1-self.ksai) * values * mask

    def cuda(self, device=None):
        DEVICE = torch.device('cuda:{}'.format(device))
        for i in range(len(self.sv)):
            self.distance_matrices[i] = self.distance_matrices[i].to(DEVICE)
            self.sv[i] = self.sv[i].to(DEVICE)
        for m in self.feature.modules():
            if isinstance(m, DropConv2d):
                m.mask = m.mask.to(DEVICE)
        return self._apply(lambda t: t.cuda(device))


class DropConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, dr=0.5):
        super(DropConv2d, self).__init__(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=bias)
        self.mask = torch.Tensor(1, in_channels // groups, 1, 1)
        self.dr = dr

    def forward(self, input):
        if self.training is True:
            index = torch.rand(1, self.in_channels, 1, 1)
            self.mask[index > self.dr] = 1.
            self.mask[index <= self.dr] = 0.
        else:
            self.mask.fill_(1-self.dr)
        return F.conv2d(input * self.mask.expand_as(input), self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



