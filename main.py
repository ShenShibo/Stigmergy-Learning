from process import *
from network import *
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import pickle
import torch


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels.data).sum()
    return correct


def validate(net, loader, use_cuda=False):
    net.test()
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


def train():
    use_cuda = True
    if torch.cuda.is_available() is False:
        use_cuda = False
    # torch.cuda.set_device(1)
    # 网络声明
    # net = NaiveNet(is_BN=False)
    net = StigmergyNet()
    if use_cuda:
        net = net.cuda()
    # 超参数设置
    epochs = 10
    lr = 0.1
    batch_size = 128

    # 参数设置
    criterion = nn.CrossEntropyLoss()
    # 自定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.00001)
    optimizer.zero_grad()
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    # 数据读入
    train_data, train_label, validate_data, validate_label = data_load()
    # 生成数据集
    train_set = MnistDataSet(train_data, train_label)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_set = MnistDataSet(validate_data, validate_label)
    validate_loader = DataLoader(val_set, batch_size=128)
    # 开始训练
    loss_save = []
    tacc_save = []
    vacc_save = []
    for epoch in range(epochs):
        lr_scheduler.step()
        running_loss = 0.0
        correct_count = 0.
        count = 0
        for i, (b_x, b_y) in enumerate(train_loader):
            size = b_x.shape[0]
            b_x = Variable(b_x)
            b_y = Variable(b_y)
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            #
            outputs = net(b_x)

            optimizer.zero_grad()
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()

            # 计算loss
            running_loss += loss.item()
            count += size
            correct_count += accuracy(outputs, b_y).item()
            if (i + 1) % 15 == 0:
                acc = validate(net, validate_loader, use_cuda)
                print('[ %d-%d ] loss: %.9f, \n'
                      'training accuracy: %.6f, \n'
                      'validating accuracy: %.6f' % (
                      epoch + 1, i + 1, running_loss / count, correct_count / count, acc))
                tacc_save.append(correct_count / count)
                loss_save.append(running_loss / count)
                vacc_save.append(acc)
        if (epoch + 1) % 1 == 0:
            print("save")
            torch.save(net.state_dict(), './model/dropout_net{}.p'.format(epoch + 1))
    dic = {}
    dic['loss'] = loss_save
    dic['training_accuracy'] = tacc_save
    dic['validating_accuracy'] = vacc_save
    with open('./model/record.p', 'wb') as f:
        pickle.dump(dic, f)


if __name__ == "__main__":
    train()
