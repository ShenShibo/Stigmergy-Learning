# -*-coding:utf-8-*-
import numpy as np
import struct
from torch.utils.data.dataset import Dataset
import random
import cv2
import pickle

tlabel_path = './data/MNIST/train-labels.idx1-ubyte'
tdata_path = './data/MNIST/train-images.idx3-ubyte'
test_label_path = './data/MNIST/t10k-labels.idx1-ubyte'
test_data_path = './data/MNIST/t10k-images.idx3-ubyte'
# 数据读取
def data_load():
    # 读取训练
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

def cifar_load(path_list = []):
    length = 32 * 32
    train_data = None
    train_label = None
    val_data = None
    val_label = None
    for l in path_list:
        print(l)
        with open('./data/cifar-10-python/{}'.format(l), 'rb') as f:

            samples = pickle.load(f, encoding='bytes')
            data = []
            # raw_img = np.zeros((32, 32, 3), dtype=np.uint8)
            val = []
            for i, d in enumerate(samples[b'data']):
                # 随机采样，平移
                r = np.reshape(d[:length], (32, 32))
                g = np.reshape(d[length:2 * length], (32, 32))
                b = np.reshape(d[-length:], (32, 32))
                raw = np.array([b, g, r], dtype=np.uint8)
                if i >= 9000:
                    val.append(raw)
                    continue
                # 原始数据
                data.append(raw)
                padding1 = np.zeros((2, 32), dtype=np.uint8)
                padding2 = np.zeros((36, 2), dtype=np.uint8)

                temp_r = np.concatenate((r, padding1), axis=0)
                temp_r = np.concatenate((padding1, temp_r), axis=0)
                temp_r = np.concatenate((temp_r, padding2), axis=1)
                temp_r = np.concatenate((padding2, temp_r), axis=1)

                temp_g = np.concatenate((g, padding1), axis=0)
                temp_g = np.concatenate((padding1, temp_g), axis=0)
                temp_g = np.concatenate((temp_g, padding2), axis=1)
                temp_g = np.concatenate((padding2, temp_g), axis=1)

                temp_b = np.concatenate((b, padding1), axis=0)
                temp_b = np.concatenate((padding1, temp_b), axis=0)
                temp_b = np.concatenate((temp_b, padding2), axis=1)
                temp_b = np.concatenate((padding2, temp_b), axis=1)

                for j in range(4):
                    rd_x = random.randint(16, 20)
                    rd_y = random.randint(16, 20)
                    if rd_x == 18 and rd_y == 18:
                        rd_x = random.randint(16, 20)
                        rd_y = random.randint(16, 20)
                    r = temp_r[rd_y - 16:rd_y + 16, rd_x - 16:rd_x + 16]
                    g = temp_g[rd_y - 16:rd_y + 16, rd_x - 16:rd_x + 16]
                    b = temp_b[rd_y - 16:rd_y + 16, rd_x - 16:rd_x + 16]
                    data.append(np.array([b, g, r], dtype=np.uint8))
                # 映射
                rotation1 = raw[:, ::-1, :]
                rotation2 = raw[:, :, ::-1]
                data.append(rotation1)
                data.append(rotation2)
                # 尺度
                # up_scale = 1.1
                # up_sample = np.zeros((3, 32, 32), dtype=np.uint8)
                # down_sample = np.zeros((3, 32, 32), dtype=np.uint8)
                # down_scale = 0.9
                # temp = cv2.resize(raw_img,
                #                   (0, 0),
                #                   fx=up_scale,
                #                   fy=up_scale,
                #                   interpolation=cv2.INTER_CUBIC)
                # for k in range(3):
                #     up_sample[k,:,:] = temp[
                #                 temp.shape[1]/2-16:temp.shape[1]/2+16,
                #                 temp.shape[0]/2-16:temp.shape[0]/2+16,
                #                 k]
                #
                # temp = cv2.resize(raw_img, (0, 0), fx=down_scale, fy=down_scale)
                # for k in range(3):
                #     down_sample[
                #     k,
                #     16-(temp.shape[1]+1)/2:16+temp.shape[1]/2,
                #     16-(temp.shape[0]+1)/2:16+temp.shape[0]/2] = temp[:,:,k]
                # data.append(up_sample)
                # data.append(down_sample)
                # # 加椒盐噪声
                # noise = raw.copy()
                # for k in range(25):
                #     rx = random.randint(0, 31)
                #     ry = random.randint(0, 31)
                #     noise[:, rx, ry] = [255, 255, 255]
                # for k in range(25):
                #     rx = random.randint(0, 31)
                #     ry = random.randint(0, 31)
                #     noise[:, rx, ry] = [0, 0, 0]
                # data.append(noise)

            data = np.array(data)
            labels = np.array(samples[b'labels'][:9000]).repeat(7)
            val = np.array(val)
            vlabels = np.array(samples[b'labels'][-1000:])
        if l == 'data_batch_1':
            train_data = data.copy()
            train_label = labels.copy()
            val_data = val.copy()
            val_label = vlabels.copy()
        else:
            train_data = np.concatenate((train_data, data), axis=0)
            train_label = np.concatenate((train_label, labels), axis=0)
            val_data = np.concatenate((val_data, val), axis=0)
            val_label = np.concatenate((val_label, vlabels), axis=0)
        print(train_data.shape)
        print(train_label.shape)
        print(val_data.shape)
        print(val_label.shape)
    return train_data, train_label, val_data, val_label


class CifarDataSet(Dataset):
    def __init__(self, data, labels):
        super(CifarDataSet, self).__init__()
        # 数据打散
        print("Data set is creating...")
        self.data = data.astype(np.float32)
        self.label = labels
        self.data /= 255.
        print("Done!")

    def __getitem__(self, item):
        return self.data[item, :, :, :], self.label[item]

    def __len__(self):
        return self.data.shape[0]