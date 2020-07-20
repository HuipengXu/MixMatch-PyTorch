from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.textcnn import MixTextCNN, Config
from dataset.imdb import get_imdb, MyIMDB
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='train batch-size')
parser.add_argument('--vocab-size', default=25000, type=int, metavar='N',
                    help='vocabulary size')
parser.add_argument('--max-length', default=512, type=int, metavar='N',
                    help='max text length')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--n-labeled', type=int, default=250,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=1024,
                    help='Number of labeled data')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing IMDB')
    train_labeled_set, train_unlabeled_set, valid_set, test_set, text_field, label_field = get_imdb('./data/aclImdb/')
    text_field.build_vocab(train_unlabeled_set, max_size=args.vocab_size)
    label_field.build_vocab(train_unlabeled_set)
    print(f"Unique tokens in TEXT vocabulary: {len(text_field.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(label_field.vocab)}")
    text_vocab, label_vocab = text_field.vocab, label_field.vocab
    train_labeled_set = MyIMDB(train_labeled_set, text_vocab, label_vocab)
    train_unlabeled_set = MyIMDB(train_unlabeled_set, text_vocab, label_vocab, unlabeled=True)
    valid_set = MyIMDB(valid_set, text_vocab, label_vocab)
    test_set = MyIMDB(test_set, text_vocab, label_vocab)

    train_labeled_loader = DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                      drop_last=True)
    train_unlabeled_loader = DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                        drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    print("==> creating TextCNN")

    def create_model(config, model=MixTextCNN, use_cuda=False, ema=False):
        model = model(config)
        if use_cuda: model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    config = Config(text_field, label_field)
    model = create_model(config, use_cuda=use_cuda)
    ema_model = create_model(config, use_cuda=use_cuda, ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'noisy-imdb'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(
            ['Train Loss', 'Train Loss X', 'Train Loss U', 'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])

    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_loss_x, train_loss_u = train(train_labeled_loader, train_unlabeled_loader, text_vocab, model,
                                                       optimizer, ema_optimizer, train_criterion, epoch, use_cuda)
        _, train_acc = validate(train_labeled_loader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_acc = validate(valid_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')

        step = args.val_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_acc, test_loss, test_acc])

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


def get_batch(iterator, loader):
    try:
        inputs, targets = next(iterator)
    except:
        iterator = iter(loader)
        inputs, targets = next(iterator)
    return iterator, inputs, targets


def truncated_padded(tensor, max_len, pad):
    bs, length = tensor.size()
    if max_len > length:
        pad_tensor = pad * torch.ones((bs, max_len - length), dtype=tensor.dtype)
        return torch.cat([tensor, pad_tensor], dim=1)
    elif max_len < length:
        return tensor[:, :max_len]
    else:
        return tensor


def train(labeled_trainloader, unlabeled_trainloader, vocab, model, optimizer, ema_optimizer, criterion, epoch,
          use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        labeled_train_iter, inputs_x, targets_x = get_batch(labeled_train_iter, labeled_trainloader)
        unlabeled_train_iter, inputs_u, inputs_u2 = get_batch(unlabeled_train_iter, unlabeled_trainloader)

        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 2).scatter_(1, targets_x.view(-1, 1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        length_x, length_u, length_u2 = inputs_x.size(1), inputs_u.size(1), inputs_u2.size(1)
        pad = vocab.stoi['<pad>']
        max_len = max(length_x, length_u, length_u2, args.max_length)
        inputs_x, inputs_u, inputs_u2 = [truncated_padded(inputs, max_len, pad)
                                         for inputs in [inputs_x, inputs_u, inputs_u2]]
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1 - l)
        # shuffle
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        # 13、14行合成一步了
        mixed_target = l * target_a + (1 - l) * target_b
        logits_x, logits_u = model(input_a, input_b, l, training=True)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                              epoch + batch_idx / args.val_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
            batch=batch_idx + 1,
            size=args.val_iteration,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            w=ws.avg,
        )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg,)


def validate(valloader, model, criterion, epoch, use_cuda, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            # 注意和训练的时候计算方式有些不一样
            loss = criterion(outputs[:, 1], targets)

            # measure accuracy and record loss
            acc = accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            Acc.update(acc.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                acc=Acc.avg,
            )
            bar.next()
        bar.finish()
    return (losses.avg, Acc.avg)


def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        # 标注数据损失
        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lx = nn.BCEWithLogitsLoss()(outputs_x, targets_x)
        # 无标注数据损失
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, args.lambda_u * linear_rampup(epoch)  # 线性增大 lambda_u


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype != torch.float32: continue
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)


if __name__ == '__main__':
    main()
