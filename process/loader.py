# -*-coding:utf-8-*-
import numpy as np
import struct
from torch.utils.data.dataset import Dataset

tlabel_path = '../data/MNIST/train-labels.idx1-ubyte'
tdata_path = '../data/MNIST/train-images.idx3-ubyte'
test_label_path = '../data/MNIST/t10k-labels.idx1-ubyte'
test_data_path = '../data/MNIST/t10k-images.idx3-ubyte'
# 数据读取
def data_load():
    # 读取训练数据

    with open(tlabel_path, 'rb') as f:
        magic, n = struct.unpack('>II',
                                 f.read(8))
        tlabels = np.fromfile(f, dtype=np.uint8)
    with open(tdata_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               f.read(16))
        tdata = np.fromfile(f,
                             dtype=np.uint8).reshape(len(tlabels), 1, 28, 28)
    # 读取测试数据
    with open(test_label_path, 'rb') as f:
        magic, n = struct.unpack('>II',
                                 f.read(8))
        test_labels = np.fromfile(f, dtype=np.uint8)
    with open(test_data_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               f.read(16))
        test_data = np.fromfile(f,
                            dtype=np.uint8).reshape(len(test_labels), 1, 28, 28)
    return tdata, tlabels, test_data, test_labels


class MnistDataSet(Dataset):
    def __init__(self, data, labels):
        super(MnistDataSet, self).__init__()
        # 数据打散
        print("Data set is creating...")
        self.data = data
        self.label = labels.astype(np.int64)
        print("Data load done!")

    def __getitem__(self, item):
        return self.data[item, :, :, :].astype(np.float32)/255., self.label[item]

    def __len__(self):
        return self.data.shape[0]