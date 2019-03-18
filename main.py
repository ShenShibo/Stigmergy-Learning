from process import *
from network import *
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
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
        if use_cuda:
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        outputs = net(b_x)
        c = accuracy(outputs, b_y)
        correct_count += c
        count += size
    acc = correct_count.item() / float(count)
    return acc


def train(args=None):
    assert args is not None
    use_cuda = torch.cuda.is_available() and args.cuda
    # network declaration
    if args.network == 'Vgg':
        net = Vgg16()
        name_net = "Vgg16_cifar10_pure_0.1"
    else:
        return
    if args.pretrained:
        with open('./model/{}'.format(args.pre_model), 'rb') as f:
            net.load_state_dict(torch.load(f))
    if use_cuda:
        torch.cuda.set_device(args.cuda_device)
        net = net.cuda()
    # 超参数设置
    epochs = args.epochs
    lr = args.lr
    batch_size = args.bz
    # 误差函数设置
    criterion = nn.CrossEntropyLoss()
    # 优化器设置
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = MultiStepLR(optimizer, milestones=[75, 125], gamma=0.1)
    # 数据读入
    train_data, train_label, _, _ = cifar_load(path_list=['data_batch_1',
                                                                                   'data_batch_2',
                                                                                   'data_batch_3',
                                                                                   'data_batch_4',
                                                                                   'data_batch_5'])
    # 生成数据集
    validate_data, validate_label = cifar_load_test('test_batch')
    train_set = CifarDataSet(train_data, train_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = CifarDataSet(validate_data, validate_label)
    validate_loader = DataLoader(val_set, batch_size=256)
    # 开始训练
    loss_save = []
    tacc_save = []
    vacc_save = []
    best_acc = 0.
    dic = {}
    for epoch in range(args.start_epoch, epochs):
        running_loss = 0.0
        correct_count = 0.
        count = 0
        lr_scheduler.step()
        for i, (b_x, b_y) in enumerate(train_loader):
            size = b_x.shape[0]
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            b_x = Variable(b_x)
            b_y = Variable(b_y)
            outputs = net(b_x)
            optimizer.zero_grad()
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            # 计算loss
            running_loss += loss.item()
            count += size
            correct_count += accuracy(outputs, b_y).item()
            if (i + 1) % 100 == 0:
                print('[ %d-%d ] loss: %.9f, \n'
                      'training accuracy: %.6f' % (
                      epoch + 1, i + 1, running_loss / count,
                      correct_count / count))
                tacc_save.append(correct_count / count)
                loss_save.append(running_loss / count)
        net.train(mode=False)
        acc = validate(net, validate_loader, use_cuda)
        print('[ %d-%d]\n'
              'validating accuracy: %.6f' % (epoch+1, epochs, acc))
        vacc_save.append(acc)
        if acc > best_acc:
            best_acc = acc
            dic['best_model'] = net.state_dict()
        net.train(mode=True)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("save")
            torch.save(net.state_dict(), './model/{}_{}.p'.format(name_net, epoch + 1))
    dic['loss'] = loss_save
    dic['training_accuracy'] = tacc_save
    dic['validating_accuracy'] = vacc_save
    with open('./model/record_{}.p'.format(name_net), 'wb') as f:
        pickle.dump(dic, f)


def test(args=None):
    assert args is not None
    torch.cuda.set_device(args.cuda_device)
    use_cuda = True
    if args.network == "Vgg":
        net = Vgg16()
    with open('./model/{}'.format(args.model), 'rb') as f:
        dic = pickle.load(f)
        net.load_state_dict(dic['best_model'])
    test_data, test_labels = cifar_load_test('test_batch')
    test_set = CifarDataSet(test_data, test_labels)
    loader = DataLoader(test_set, batch_size=256)
    net.train(mode=False)
    acc = validate(net, loader, use_cuda=use_cuda)
    print("testing accuracy : {}".format(acc))
    return


def mtrain(args=None):
    assert args is not None
    use_cuda = torch.cuda.is_available() and args.cuda
    # network declaration
    net = MNISTNet()
    name_net = 'MCONV_MNIST'
    if use_cuda:
        torch.cuda.set_device(args.cuda_device)
        net = net.cuda()
    # 超参数设置
    epochs = args.epochs
    lr = args.lr
    batch_size = args.bz
    # 误差函数设置
    criterion = nn.CrossEntropyLoss()
    # 优化器设置
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = MultiStepLR(optimizer, milestones=[args.epochs // 2, 3 * args.epochs // 4], gamma=0.1)
    # 数据读入
    train_data, train_label, validate_data, validate_label = data_load()
    # 生成数据集
    train_set = MnistDataSet(train_data, train_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = MnistDataSet(validate_data, validate_label)
    validate_loader = DataLoader(val_set, batch_size=256)
    # 开始训练
    loss_save = []
    tacc_save = []
    vacc_save = []
    best_acc = 0.
    dic = {}
    for epoch in range(args.start_epoch, epochs):
        running_loss = 0.0
        correct_count = 0.
        count = 0
        lr_scheduler.step()
        for i, (b_x, b_y) in enumerate(train_loader):
            size = b_x.shape[0]
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            b_x = Variable(b_x)
            b_y = Variable(b_y)
            outputs = net(b_x)
            optimizer.zero_grad()
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            # 计算loss
            running_loss += loss.item()
            count += size
            correct_count += accuracy(outputs, b_y).item()
            if (i + 1) % 30 == 0:
                print('[ %d-%d ] loss: %.9f, \n'
                      'training accuracy: %.6f' % (
                      epoch + 1, i + 1, running_loss / count,
                      correct_count / count))
                tacc_save.append(correct_count / count)
                loss_save.append(running_loss / count)
                net.train(mode=False)
                acc = validate(net, validate_loader, use_cuda)
                print('[ %d-%d]\n'
                      'validating accuracy: %.6f' % (epoch+1, i+1, acc))
                vacc_save.append(acc)
                if acc > best_acc:
                    best_acc = acc
                    dic['best_model'] = net.state_dict()
                net.train(mode=True)
        # if (epoch + 1) % 5 == 0 or epoch == 0:
        #     print("save")
        #     torch.save(net.state_dict(), './model/{}_{}.p'.format(name_net, epoch + 1))
    dic['loss'] = loss_save
    dic['training_accuracy'] = tacc_save
    dic['validating_accuracy'] = vacc_save
    with open('./model/record_{}.p'.format(name_net), 'wb') as f:
        pickle.dump(dic, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='training or testing')
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
    parser.add_argument('--epochs', type=int, help="training epochs", default=50)
    parser.add_argument('--bz', type=int, help='batch size', default=64)
    parser.add_argument('--wd', type=float, help='weight decay', default=1e-4)
    parser.add_argument('--cuda', type=bool, help='GPU', default=True)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--network', type=str, default='Vgg')
    parser.add_argument('--model', type=str, default='record_Vgg16_cifar10_pure.p')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--pre_model', type=str, default='Vgg16_init.p')
    parser.add_argument('--start_epoch', type=int, default=0)
    # parser.add_argument('--data_set', type=str, default='cifar10')

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'mtrain':
        mtrain(args)
    else:
        test(args)
