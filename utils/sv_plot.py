import torch
from matplotlib import pyplot as plt
import pickle
import numpy as np
torch.cuda.set_device(1)
all_sv = []
for i in range(1, 16):
    with open('../model/VGG16-cifar10-stigmergy_{}.p'.format(i), 'rb') as f:
        all_sv.append(pickle.load(f)['sv'])
_dr = [.0, .1, .2, .2, .3, .3, .3, .5, .5, .5, .6, .6, .6]
val = []
for layer in range(13):
    if(layer == 0):
        continue
    X = []
    val = []
    for sv in all_sv:
        X.append(sv[layer].cpu().numpy())
    X = np.array(X).transpose()
    h, w = X.shape
    X = np.sort(X, axis=0)
    x = np.sum(X[-int((1-_dr[layer])*h):, :], axis=0)
    y = np.sum(X[:int(_dr[layer]*h), :], axis=0)
    gain = x / y
    plt.plot(gain)
    # for i in range(h):
    #     plt.plot(X[i, :])
    plt.xlabel('epochs')
    plt.ylabel('stigmergy gain')
    plt.show()

