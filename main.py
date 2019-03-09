from process import *
from network import *
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import pickle
import torch
import argparse

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels.data).sum()
    return correct


def validate(net, loader, use_cuda=False):
    correct_count = 0.
    count = 0.
    if use_cuda:
        net = net.cuda()
    for i, (b_x, b_y) in enumerate(loader, 0):
        size = b_x.shape[0]
        b_x = Variable(b_x)
        b_y = Variable(b_y)
        if use_cuda:
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        outputs = net(b_x)
        c = accuracy(outputs, b_y)
        correct_count += c
        count += size
    acc = correct_count.item() / float(count)
    return acc


def train(r=0.5):
    assert r is not None
    use_cuda = True
    if torch.cuda.is_available() is False:
        use_cuda = False
    torch.cuda.set_device(1)
    # 网络声明
    # net = NaiveNet(is_BN=False)
    # dr = 0.05
    # print("dropout rate equals {}".format(dr))
    net = Vgg16()
    name_net = "Vgg16_cifar10_pure"
    if use_cuda:
        net = net.cuda()
    # 超参数设置
    epochs = 60
    lr = 0.1
    batch_size = 128
    # 参数设置
    criterion = nn.CrossEntropyLoss()
    # 自定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # 数据读入
    # train_data, train_label, validate_data, validate_label = data_load()
    train_data, train_label, validate_data, validate_label = cifar_load(path_list=['data_batch_1',
                                                                                   'data_batch_2',
                                                                                   'data_batch_3',
                                                                                   'data_batch_4',
                                                                                   'data_batch_5'])
    # 生成数据集
    # train_set = MnistDataSet(train_data, train_label)
    train_set = CifarDataSet(train_data, train_label)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              )
    # val_set = MnistDataSet(validate_data, validate_label)
    val_set = CifarDataSet(validate_data, validate_label)
    validate_loader = DataLoader(val_set, batch_size=256)
    # 开始训练
    loss_save = []
    tacc_save = []
    vacc_save = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct_count = 0.
        count = 0
        if epoch == 30 or epoch == 45 or epoch == 54:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            optimizer.param_groups['lr'] = lr
        for i, (b_x, b_y) in enumerate(train_loader):
            size = b_x.shape[0]
            b_x = Variable(b_x)
            b_y = Variable(b_y)
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            #
            # optimizer.zero_grad()
            outputs = net(b_x)

            optimizer.zero_grad()
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()

            # 计算loss
            running_loss += loss.item()
            count += size
            correct_count += accuracy(outputs, b_y).item()
            if (i + 1) % 50 == 0:
                net.train(mode=False)
                acc = validate(net, validate_loader, use_cuda)
                print('[ %d-%d ] loss: %.9f, \n'
                      'training accuracy: %.6f, \n'
                      'validating accuracy: %.6f' % (
                      epoch + 1, i + 1, running_loss / count,
                      correct_count / count,
                      acc))
                tacc_save.append(correct_count / count)
                loss_save.append(running_loss / count)
                vacc_save.append(acc)
                net.train(mode=True)
        if (epoch + 1) % 5 == 0:
            print("save")
            torch.save(net.state_dict(), './model/{}_{}.p'.format(name_net, epoch + 1))
    dic = {}
    dic['loss'] = loss_save
    dic['training_accuracy'] = tacc_save
    dic['validating_accuracy'] = vacc_save
    with open('./model/record_{}.p'.format(name_net), 'wb') as f:
        pickle.dump(dic, f)


def test():
    torch.cuda.set_device(0)
    use_cuda = True
    net = Vgg16()
    with open('./model/Vgg16_cifar10_pure_60.p', 'rb') as f:
        net.load_state_dict(torch.load(f))
    test_data, test_labels = cifar_load_test('test_batch')
    test_set = CifarDataSet(test_data, test_labels)
    loader = DataLoader(test_set, batch_size=256)
    acc = validate(net, loader, use_cuda=use_cuda)
    print("testing accuracy : {}".format(acc))
    return

if __name__ == "__main__":

    # train()
    test()