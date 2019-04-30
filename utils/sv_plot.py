import torch
from matplotlib import pyplot as plt
import pickle
import numpy as np
torch.cuda.set_device(1)

with open('../model/record-ResNet56-0.3-no-pre-1-cifar10-ksai-0.6.p', 'rb') as f:
    model = pickle.load(f)
    loss1 = model['training_accuracy']
with open('../model/record-ResNet56-0.3-pre-1-cifar10-ksai-0.6.p', 'rb') as f:
    model = pickle.load(f)
    loss2 = model['training_accuracy']
plt.plot(range(len(loss1)), loss1)
plt.plot(range(len(loss2)), loss2)
plt.legend(['loss1', 'loss2'])
plt.show()