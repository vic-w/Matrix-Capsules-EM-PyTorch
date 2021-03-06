from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import capsules
from model.reconstruct import RECONSTRUCT
from loss import SpreadLoss
from datasets import smallNORB

import cv2
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--test-intvl', type=int, default=1, metavar='N',
                    help='test intvl (default: 1)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight-decay', type=float, default=0, metavar='WD',
                    help='weight decay (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--em-iters', type=int, default=2, metavar='N',
                    help='iterations of EM Routing')
parser.add_argument('--snapshot-folder', type=str, default='./snapshots', metavar='SF',
                    help='where to store the snapshots')
parser.add_argument('--data-folder', type=str, default='./data', metavar='DF',
                    help='where to store the datasets')
parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                    help='dataset for training(mnist, smallNORB)')


def get_setting(args):
    num_class = 2
    data_transform = transforms.Compose([
                                    transforms.Grayscale(),
                                    transforms.Resize([28,28]),
                                    transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(os.path.join('data/human/train'), data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, pin_memory=True, num_workers=0, drop_last=True, shuffle=True)
    
    test_dataset = datasets.ImageFolder(os.path.join('data/human/val'), data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, pin_memory=True, num_workers=0, drop_last=True, shuffle=True)


    return num_class, train_loader, test_loader


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, model2, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    model2.train()

    train_len = len(train_loader)
    epoch_acc = 0
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        
        # print('data.shape', data.shape)
        # print('target.shape', target.shape)

        optimizer.zero_grad()
        pos, output = model(data)

        reconstruct_out = model2(pos, output)
        # print('reconstruct_out.shape', reconstruct_out.shape)
        # print('pos.shape', pos.shape)
        # print('output.shape', output.shape)
        r = (1.*batch_idx + (epoch-1)*train_len) / (args.epochs*train_len)
        loss1 = criterion(output, target, r)
        #norm_data = ((data+0.43)/3.3)
        #print(reconstruct_out.shape)
        #print(target.shape)
        mean_diff = torch.mean(torch.pow(reconstruct_out-data, 2), dim=(1,2,3))
        is_face_float = torch.tensor((1-target), dtype=torch.float).cuda()
        mean_diff = mean_diff * is_face_float
        mean_diff /= torch.mean(is_face_float)

        #print(mean_diff.shape)
        loss2 = torch.mean(mean_diff)
        #print(loss2.item())
        loss = loss1 + loss2*10
        acc = accuracy(output, target)
        loss.backward()
        optimizer.step()

        data_show = np.array(data.cpu().detach().numpy()[0,:,:,:].reshape([28,28])*255, dtype=np.uint8)
        recon_show = np.array(reconstruct_out.cpu().detach().numpy()[0,:,:,:].reshape([28,28])*255, dtype=np.uint8)
        cv2.namedWindow('data', 0)
        cv2.namedWindow('recon', 0)
        cv2.imshow('data', data_show)
        cv2.imshow('recon', recon_show)
        cv2.waitKey(1)


        batch_time.update(time.time() - end)
        end = time.time()

        epoch_acc += acc[0].item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}\tAccuracy: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  loss.item(), acc[0].item(),
                  batch_time=batch_time, data_time=data_time))
    return epoch_acc


def snapshot(model, model2, folder, epoch):
    path = os.path.join(folder, 'face_model_{}.pth'.format(epoch))
    path2 = os.path.join(folder, 'face_model2_{}.pth'.format(epoch))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print('saving model to {}'.format(path))
    torch.save(model.state_dict(), path)
    torch.save(model2.state_dict(), path2)


def test(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for data, target in test_loader:
            #print(data.shape, target.shape)
            data, target = data.to(device), target.to(device)
            pos,output = model(data)
            test_loss += criterion(output, target, r=1).item()
            acc += accuracy(output, target)[0].item()

    test_loss /= test_len
    acc /= test_len
    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, acc))
    return acc


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    # datasets
    num_class, train_loader, test_loader = get_setting(args)

    # model
    A, B, C, D = 64, 8, 16, 16
    # A, B, C, D = 32, 32, 32, 32
    model = capsules(A=A, B=B, C=C, D=D, E=num_class,
                     iters=args.em_iters).to(device)
    model2 = RECONSTRUCT(num_class=num_class).to(device)

    #model.load_state_dict(torch.load('snapshots/model_1.pth'))
    #model2.load_state_dict(torch.load('snapshots/model2_1.pth'))

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)
    #print(model.parameters())
    #print(model2.parameters())
    params = list(model.parameters()) + list(model2.parameters())    
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    best_acc = test(test_loader, model, criterion, device)
    for epoch in range(1, args.epochs + 1):
        acc = train(train_loader, model, model2, criterion, optimizer, epoch, device)
        acc /= len(train_loader)
        scheduler.step(acc)
        if epoch % args.test_intvl == 0:
            best_acc = max(best_acc, test(test_loader, model, criterion, device))

        snapshot(model, model2, args.snapshot_folder, epoch)

    best_acc = max(best_acc, test(test_loader, model, criterion, device))
    print('best test accuracy: {:.6f}'.format(best_acc))

    snapshot(model, model2, args.snapshot_folder, args.epochs)

if __name__ == '__main__':
    main()
