import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
# import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from arch import EnsembleNet
from TinyImageNetDataLoader import TrainTinyImageNetDataset, TestTinyImageNetDataset, id_dict



parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--train_bs', default=64, type=int)
parser.add_argument('--test_bs', default=100, type=int)
parser.add_argument('--pp', default=5, type=int, help='partition point')
parser.add_argument('--ep', default=100, type=int, help='epochs')
parser.add_argument('--nu', default=5, type=int, help='num of users')
parser.add_argument('--id', default=0, type=int, help='index of users')
parser.add_argument('--n_embed', default=4096, type=int, help='embedding size')
parser.add_argument('--n_parts', default=8, type=int, help='number of parts')
parser.add_argument('--commitment', default=1.0, type=int, help='commitment')
parser.add_argument('--skip_quant', action='store_true', help='skip quantization')
parser.add_argument('--quant', default='VQ', help='number of parts')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()


# Training
def train(ep, resume):
    print('\nEpoch: %d' % ep)
    if args.resume:
        net.decoder.train()
    else:
        net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # if len(inputs) != args.train_bs:
        #     continue
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / len(trainloader), 100. * correct / total, correct, total))


def test(ep):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / len(testloader), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        if args.skip_quant:
            state = {
                # 'encoder': net.encoder.state_dict(),
                # 'decoder': net.decoder.state_dict(),
                'net': net.state_dict(),
                'acc': acc,
                'epoch': ep,
            }
        else:
            if args.id == -1:
                state = {
                    'encoder': net.encoder.state_dict(),
                    'quantizer': net.quantizer.state_dict(),
                    'decoder': net.decoder.state_dict(),
                    'acc': acc,
                    'epoch': ep,
                }
            else:
                state = {
                    'decoder': net.decoder.state_dict(),
                    'acc': acc,
                    'epoch': ep,
                }
        torch.save(state, f'{exp_path}/{args.id}.pth')
        best_acc = acc

    return acc


if __name__ == '__main__':

    if args.skip_quant:
        exp_name = f'{args.lr}_{args.ep}_{args.skip_quant}_Tiny_AdaptE'
    else:
        exp_name = (f'{args.lr}_{args.pp}_{args.ep}_{args.nu}_{args.n_embed}_{args.n_parts}_'
                    f'{args.commitment}_{args.quant}_AdaptE')
    print(args)

    exp_path = f'./checkpoint/ckpt_{exp_name}/'
    os.makedirs(exp_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))

    trainset = TrainTinyImageNetDataset(id=id_dict, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=2)
    testset = TestTinyImageNetDataset(id=id_dict, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, num_workers=1)

    if args.id == -1:
        if args.skip_quant:
            # shared_idx = 35000 if args.nu == 16 else 42000
            shared_idx = 100000
        else:
            shared_idx = 34000 if args.nu == 16 else 40000
        shared_indices = np.arange(0, shared_idx)
        shared_set = torch.utils.data.Subset(trainset, shared_indices)
        trainloader = torch.utils.data.DataLoader(
            shared_set, batch_size=args.train_bs, shuffle=True, num_workers=1)
    else:
        if args.nu == 16:
            shared_idx = 34000
            shared_indices = np.arange(0, shared_idx)
            dataloaders = []
            for i in range(args.nu):
                part_indices = np.arange(shared_idx + i * 1000, shared_idx + i * 1000 + 1000)
                part_set = torch.utils.data.Subset(trainset, np.concatenate((shared_indices, part_indices)))
                dataloaders.append(torch.utils.data.DataLoader(part_set, batch_size=args.train_bs,
                                                               shuffle=True, num_workers=1, drop_last=False))
        elif args.nu == 5:
            shared_idx = 40000
            shared_indices = np.arange(0, shared_idx)
            dataloaders = []
            for i in range(args.nu):
                part_indices = np.arange(shared_idx + i * 2000, shared_idx + i * 2000 + 2000)
                part_set = torch.utils.data.Subset(trainset, np.concatenate((shared_indices, part_indices)))
                dataloaders.append(torch.utils.data.DataLoader(part_set, batch_size=args.train_bs,
                                                               shuffle=True, num_workers=1, drop_last=False))

        trainloader = dataloaders[args.id]

        assert len(dataloaders) == args.nu

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')

    # net = EnsembleNet(res_stop=args.pp, ncls=100, skip_quant=args.skip_quant, quant=args.quant,
    #                   n_embed=args.n_embed, n_parts=args.n_parts, commitment=args.commitment)

    mobilenet_v2 = models.mobilenet_v2(num_classes=1000, width_mult=1.0)
    mobilenet_v2.load_state_dict(torch.load('mobilenet_v2-b0353104.pth'), strict=True)

    net_layers = []

    X = torch.rand(size=(2, 3, 64, 64))

    for layer_idx, l in enumerate(mobilenet_v2.features):
        X = l(X)
        print(l.__class__.__name__, 'Output shape:\t', X.shape)
        net_layers.append(l)

    dropout = nn.Dropout(0.2, inplace=True)
    fc = nn.Linear(in_features=1280, out_features=200, bias=True)
    classifier = nn.Sequential(dropout, fc)
    pool = nn.AdaptiveAvgPool2d(1)
    net_layers.append(pool)
    net_layers.append(nn.Flatten())
    net_layers.append(classifier)

    net = nn.Sequential(*net_layers)

    X = torch.rand(size=(2, 3, 32, 32))
    print(net(X).shape)
    net = net.to(device)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'{exp_path}/-1.pth')

        net.encoder.load_state_dict(checkpoint['encoder'])
        for param in net.encoder.parameters():
            param.requires_grad = False
        net.encoder.eval()

        if not args.skip_quant:
            net.quantizer.load_state_dict(checkpoint['quantizer'])
            for param in net.quantizer.parameters():
                param.requires_grad = False
            net.quantizer.eval()

        # net.decoder.load_state_dict(checkpoint['decoder'])

        best_acc = 0
        start_epoch = 0

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd) scheduler =
    # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) scheduler =
    # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.ep * 3 / 7.), int(args.ep * 5 / 7.)],
    # gamma=0.1)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.id == -1:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    acc_list = []
    # print(test(0))
    for epoch in range(start_epoch, start_epoch + args.ep):
        train(epoch, args.resume)
        scheduler.step()
        acc_list.append(test(epoch))

    print(best_acc)

    plt.plot(np.arange(args.ep), acc_list)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig(f'{exp_path}/{args.id}_{best_acc:.3f}.png')
