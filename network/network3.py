import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward1(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return self.relu(out)

    def forward2(self, x):
        out = self.conv2(x)
        out = self.bn2(out)
        return self.relu(out)

    def add_residual(self, x, y):
        y = self.conv3(y)
        y = self.bn3(y)
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        return self.relu(y)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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


class SResNet(nn.Module):
    # 50 layers resnet for ImageNet dataset
    _layer = [0.4, 0.4, 0.4, 0.4]

    def __init__(self, num_classes=1000, update_round=1, is_stigmergy=True, ksai=0.8):
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
        self.inplanes=64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        block = Bottleneck
        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.layer3 = self._make_layer(block, 256, 6, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_para(self, rounds=1, planes=64):
        for i in range(rounds):
            self.distance_matrices.append(torch.eye(planes))
            self.counter.append(torch.zeros(planes, planes))
            self.sv.append(torch.zeros(planes))
            self.mask.append(torch.zeros(1, planes, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self._make_para(rounds=2, planes=planes)
        self.inplanes = planes * block.expansion
        # self._make_para(rounds=1, planes=planes)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            self._make_para(rounds=2, planes=planes)

        return nn.Sequential(*layers)

    def forward(self, x, iterations):
        count = 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for l, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
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
                    self.mask[count][:, index, :, :] = 1.
                # TODO:skip sensitive layers
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
                    self.mask[count][:, index, :, :] = 1.
                # TODO:skip sensitive layers
                x = x * self.mask[count].expand_as(x)
                count += 1
                x = m.add_residual(residual, x)
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
