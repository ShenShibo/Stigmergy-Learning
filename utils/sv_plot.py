import torch
from matplotlib import pyplot as plt
import pickle
import numpy as np
torch.cuda.set_device(1)
all_sv = []
for i in range(5, 35, 5):
    with open('../model/Vgg16-cifar10-{}-ksai-0.9.p'.format(i), 'rb') as f:
        all_sv.append(pickle.load(f)['sv'])

val = []
for layer in range(13):
    X = []
    val = []
    for sv in all_sv:
        X.append(sv[layer].cpu().numpy())
    X = np.array(X).transpose()
    h, w = X.shape
    # X = np.sort(X, axis=0)
    # x = np.sum(X[-int((1-_dr[layer])*h):, :], axis=0)
    # y = np.sum(X[:int(_dr[layer]*h), :], axis=0)
    # gain = x / y
    # plt.plot(gain)
    for i in range(h):
        plt.plot(X[i, :])
    plt.xlabel('epochs')
    plt.ylabel('stigmergy gain')
    plt.show()

