'''Train CIFAR10 with PyTorch and save per-epoch metrics.'''
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def train(epoch, net, trainloader, criterion, optimizer, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # CrossEntropyLoss returns a batch mean; weight it by batch size.
        train_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/total, 100.*correct/total, correct, total))

    return train_loss / total, 100. * correct / total


def test(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/total, 100.*correct/total, correct, total))

    return test_loss / total, 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=200, type=positive_int,
                        help='number of epochs to run in this invocation')
    parser.add_argument('--batch-size', default=128, type=positive_int,
                        help='training batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='SGD weight decay')
    parser.add_argument('--log-dir', default='runs', help='parent directory for experiment logs')
    parser.add_argument('--run-name', default=None, help='experiment label used in plots')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume model weights from checkpoint/ckpt.pth')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0)

    # Model: uncomment another model here to compare architectures.
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    model_name = net.__class__.__name__
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # As in the original project, resume weights, not optimizer state.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=device)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # A separate directory per invocation prevents experiments overwriting logs.
    run_dir = Path(args.log_dir) / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), model=model_name, device=device,
                  start_epoch=start_epoch, optimizer='SGD',
                  scheduler='CosineAnnealingLR', test_batch_size=100)
    with (run_dir / 'config.json').open('w', encoding='utf-8') as config_file:
        json.dump(config, config_file, indent=2, ensure_ascii=False)
    print('==> Saving experiment logs to: %s' % run_dir)

    # Open once, then flush each completed epoch so partial runs remain usable.
    with (run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=[
            'epoch', 'lr', 'train_loss', 'train_acc', 'test_loss', 'test_acc',
        ])
        writer.writeheader()
        metrics_file.flush()
        for epoch in range(start_epoch, start_epoch + args.epochs):
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc = train(epoch, net, trainloader, criterion, optimizer, device)
            test_loss, test_acc = test(net, testloader, criterion, device)
            writer.writerow(dict(epoch=epoch, lr=lr, train_loss=train_loss,
                                 train_acc=train_acc, test_loss=test_loss, test_acc=test_acc))
            metrics_file.flush()

            # Preserve the original best-checkpoint location and file format.
            if test_acc > best_acc:
                print('Saving..')
                state = {'net': net.state_dict(), 'acc': test_acc, 'epoch': epoch}
                os.makedirs('checkpoint', exist_ok=True)
                torch.save(state, './checkpoint/ckpt.pth')
                best_acc = test_acc
            scheduler.step()


if __name__ == '__main__':
    main()
