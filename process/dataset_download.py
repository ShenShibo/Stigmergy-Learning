import torch
from torchvision.datasets import cifar

cifar.CIFAR10(root='../data/cifar-10', train=True, download=True,)