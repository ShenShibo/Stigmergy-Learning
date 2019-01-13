# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F


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

class DropoutNet(nn.Module):
    def __init__(self, p=0.5):
        super(DropoutNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=p, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(8 * 8 * 128, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def test(self):
        self.dropout1.training = False
        self.dropout2.training = False

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=1)



class StigmergyNet(nn.Module):
    def __init__(self):
        super(StigmergyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, *input):
        pass




