# -*-coding:utf-8-*-

from network import *
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import pickle
import torch
import argparse
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10 as dataset
import copy
from process import *


def train(args=None):
    assert args is not None
    use_cuda = torch.cuda.is_available() and args.cuda
    # network declaration
    if args.network == 'Vgg':
        print("Vgg")
        print("diffusion:{}, decay:{}".format(args.diffusion, args.decay))
        net = Svgg(num_classes=10, is_stigmergy=args.stigmergy, decay=args.decay, diffusion=args.diffusion)
    elif args.network == 'ResNet':
        print("ResNet")
        net = SResNet(num_classes=10, update_round=1, is_stigmergy=args.stigmergy, ksai=args.ksai)
    else:
        return
    name_net = args.name
    if args.pretrained:
        with open('./model/{}'.format(args.pre_model), 'rb') as f:
            dic3 = pickle.load(f)
            net.load_state_dict(dic3['model'])
    if use_cuda:
        torch.cuda.set_device(args.cuda_device)
        net.cuda(device=args.cuda_device)
    # 超参数设置
    epochs = args.epochs
    lr = args.lr
    batch_size = args.bz
    # 误差函数设置
    criterion = nn.CrossEntropyLoss()
    # 优化器设置
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=args.wd, nesterov=False)
    lr_scheduler = MultiStepLR(optimizer, milestones=[100, 150, 180], gamma=0.1)

    # 数据读入
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 生成数据集
    train_set = dataset(root='./data', train=True, download=False, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers)
    val_set = dataset(root='./data', train=False, download=False, transform=transform_test)
    validate_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=args.workers)

    # 开始训练
    loss_save = []
    tacc_save = []
    vacc_save = []
    best_acc = 0.
    dic = {}
    dic2 = {}
    net.stigmergy = False
    for epoch in range(args.start_epoch, epochs):
        running_loss = 0.0
        correct_count = 0.
        count = 0
        lr_scheduler.step()
        if (epoch+1) == 10:
            net.stigmergy = True
        for i, (b_x, b_y) in enumerate(train_loader):
            size = b_x.shape[0]
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
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
                print(
                    '[ %d-%d ] loss: %.9f, \n'
                    'training accuracy: %.6f' % (
                    epoch+1, i + 1, running_loss / count,
                    correct_count / count))
                tacc_save.append(correct_count / count)
                loss_save.append(running_loss / count)
        if epoch == 0:
            print("save")
            dic2['cu'] = net.channel_utility
            dic2['rm'] = net.relevance_matrices
            dic2['model'] = net.state_dict().copy()
            with open('./model/{}-{}.p'.format(name_net, epoch + 1), 'wb') as f:
                pickle.dump(dic2, f)
        net.train(mode=False)
        acc = validate(net, validate_loader, use_cuda, device=args.cuda_device)
        print('[ %d-%d]\n'
              'validating accuracy: %.6f' % (epoch+1, epochs, acc))
        vacc_save.append(acc)
        if acc > best_acc:
            best_acc = acc
            dic['best_model'] = copy.deepcopy(net.state_dict())
            dic['best_cu'] = copy.deepcopy(net.channel_utility)
            dic['best_rm'] = copy.deepcopy(net.relevance_matrices)
        net.train(mode=True)
    dic['loss'] = loss_save
    dic['training_accuracy'] = tacc_save
    dic['validating_accuracy'] = vacc_save
    with open('./model/record-{}-diffusion-{}-decay-{}.p'.format(name_net, args.diffusion, args.decay), 'wb') as f:
        pickle.dump(dic, f)


def test(args=None):
    assert args is not None
    torch.cuda.set_device(args.cuda_device)
    use_cuda = True
    # net = Vgg()
    net = Svgg()
    # net = SResNet()
    with open('./model1/record-Vgg16-0.7-pre-2-cifar10-ksai-0.6.p', 'rb') as f:
        dic = pickle.load(f)
        net.load_state_dict(dic['best_model'])
        net.sv = dic['best_sv']
        net.distance_matrices = dic['best_dm']
        print(max(dic['validating_accuracy']), dic['validating_accuracy'])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_set = dataset(root='./data', train=False, download=False, transform=transform_test)
    loader = DataLoader(val_set, batch_size=256, shuffle=True, num_workers=args.workers)
    net.eval()
    acc = validate(net, loader, use_cuda=use_cuda, device=args.cuda_device)
    print("testing accuracy : {}".format(acc))
    return


if __name__ == "__main__":
    net = "VGG-0.5"
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='training or testing')
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
    parser.add_argument('-decay', type=float, default=0.8, help='decay factor in the evaporation process')
    parser.add_argument('-diffusion', type=float, default=0.5, help='diffusion factor in the diffusion process')
    parser.add_argument('--epochs', type=int, help="training epochs", default=200)
    parser.add_argument('--bz', type=int, help='batch size', default=64)
    parser.add_argument('--wd', type=float, help='weight decay', default=1e-4)
    parser.add_argument('--cuda', type=bool, help='GPU', default=True)
    parser.add_argument('-cuda_device', type=int, default=1)
    parser.add_argument('--network', type=str, default='Vgg')
    parser.add_argument('--model', type=str, default='record-{}.p'.format(net))
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--pre_model', type=str, default='VGG-0.5-1.p'.format(net))
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-name', type=str, default='{}'.format(net))
    parser.add_argument('--stigmergy', type=bool, default=True)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        test(args)
