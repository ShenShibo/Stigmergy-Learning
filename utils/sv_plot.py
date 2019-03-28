import torch
from matplotlib import pyplot as plt
import pickle

all_sv = []
for i in range(5, 150, 5):
    with open('VGG16-cifar10-stigmergy_{}'.format(i), 'rb') as f:
        all_sv.append(pickle.load(f)['sv'])

layer=5
X = []
for sv in all_sv:
    X.append(sv[layer].cpu().numpy())

plt.figure(figsize=(10, 8))
plt.xlabel('filters')
plt.ylabel('input channel')
plt.imshow(X,
           aspect='equal',
           cmap='hot',
           origin='low')
plt.colorbar()
plt.title("Conv2 layer{} state value".format(layer))
plt.show()