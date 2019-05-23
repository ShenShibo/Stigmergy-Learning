import torch
from matplotlib import pyplot as plt
import pickle
import numpy as np
torch.cuda.set_device(1)

with open('../model/record-ResNet56-0.3-2-cifar10-ksai-0.6.p', 'rb') as f:
    model = pickle.load(f)
    dm1 = model['best_dm']
# with open('../model/record-Vgg16-0.3-pre-2-cifar10-ksai-0.6.p', 'rb') as f:
#     model = pickle.load(f)
#     loss2 = model['best_dm']
for i in range(len(dm1)):
    dm = dm1[i].cpu().numpy()
    dm[np.eye(dm.shape[0], dm.shape[1]) == 1] = 0.
    dm = np.reshape(dm, dm.shape[0] * dm.shape[1])

    plt.scatter(range(dm.shape[0]), dm)
    plt.show()
