from matplotlib import pyplot as plt
import torch
from network import *
import pickle
import numpy as np

net = VGG()
record = "../model/record_VGG16-cifar10-parameter.p"

with open(record, 'rb') as f:
    record = pickle.load(f)
    net.load_state_dict(record['best_model'])

count = 0
for module in net.modules():
    print(module)
    if isinstance(module, LmaskConv2d):
        count += 1
        weight = module.fMask.numpy()
        print(weight.shape)