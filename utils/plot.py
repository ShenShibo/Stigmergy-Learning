from matplotlib import pyplot as plt
import torch
from network import *
import pickle
import numpy as np

net = VGG(method=0)
record = "../model/record_Vgg16_cifar10_ALL0.5.p"

with open(record, 'rb') as f:
    record = pickle.load(f)
    net.load_state_dict(record['best_model'])

count = 1
for module in net.modules():
    if isinstance(module, LmaskConv2d):
        weight = module.fMask.detach().numpy()
        o, i, h, w = weight.shape
        X = weight.reshape(o, i).transpose()

        plt.figure(figsize=(10, 8))
        plt.xlabel('filters')
        plt.ylabel('input channel')
        plt.imshow(X,
                   aspect='equal',
                   interpolation='nearest',
                   cmap='hot',
                   origin='low')
        plt.colorbar()
        plt.title("Conv2 layer{} fMask value".format(count))
        plt.show()
        count += 1
    if isinstance(module, MaskConv2d):
        weight = torch.norm(module.weight.data, dim=(2, 3)).numpy()
        print(weight.shape)
        X = weight.transpose()

        plt.figure(figsize=(10, 8))
        plt.xlabel('filters')
        plt.ylabel('input channel')
        plt.imshow(X,
                   aspect='equal',
                   interpolation='nearest',
                   cmap='hot',
                   origin='low')
        plt.colorbar()
        plt.title("Conv2 layer{} filter weight norm".format(count))
        plt.show()
        count += 1