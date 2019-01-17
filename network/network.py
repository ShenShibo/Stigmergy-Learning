# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch._jit_internal import weak_script_method, weak_module
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
    def __init__(self):
        super(StigmergyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cMask = torch.Tensor(1, 128, 1, 1)
        self.cMask.fill_(.5)
        self.state_value = torch.ones(128)
        self.state_value = F.softmax(self.state_value, dim=0)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

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

    def test(self):
        self.training = False

    def _dropout(self):
        p = 0.5
        dim = 128
        if self.training:
            self.cMask.fill_(1.)
            # 随机选择一半特征图置0
            index = torch.randperm(dim)[:dim//2]
            self.cMask[:, index, :, :] = 0.
            # print(self.cMask.view(128))
        else:
            self.cMask.fill_(p)

    def _stigmergy(self):
        dim = 128
        if self.training is False:
            return
        wg = self.conv2.weight.grad
        if wg is None:
            self.cMask.fill_(0.)
            index = torch.randperm(dim)[:dim // 2]
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
            median = torch.median((-temp_if).exp())
            # 减去中值(也可以减去均值)
            influence_values = (-influence_values).exp() - median
            # 乘以掩码，去掉未参与通道数
            influence_values = self.cMask.view(128) * influence_values
            self.state_value = F.softmax(self.state_value + influence_values, dim=0)
            # 升序排列，排列好的数值在原先tensor中的索引
            index = torch.argsort(self.state_value, descending=True)[:dim//2]
            # 生成掩码
            self.cMask.fill_(0.)
            self.cMask[:, index, :, :] = 1.

    def cuda(self, device=None):
        self.state_value = self.state_value.cuda(device=device)
        self.cMask = self.cMask.cuda(device=device)
        return self._apply(lambda t: t.cuda(device))



