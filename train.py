import argparse
import os
import shutil
import time
import random
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe

from progress.bar import Bar
from tensorboardX import SummaryWriter

from textcnn import MixTextCNN, Config
from preprocess import get_imdb
from util import mkdir_p, AverageMeter, Logger, accuracy, \
    SemiLoss, WeightEMA, save_checkpoint, MyIMDB


parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='train batch-size')
parser.add_argument('--vocab-size', default=50000, type=int, metavar='N',
                    help='vocabulary size')
parser.add_argument('--max-length', default=512, type=int, metavar='N',
                    help='max text length')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--pad-token', default='<pad>', type=str,
                    help='token used for pad sentence to max length')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manual-seed', type=int, default=None, help='manual seed')
# Device options
parser.add_argument('--gpus', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--n-labeled', type=int, default=1000,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=1024,
                    help='Number of labeled data')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=0.3, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)


def set_seed(args):
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.n_gpus > 0:
        torch.cuda.manual_seed_all(args.manual_seed)

def main(args):
    best_acc = 0

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    use_cuda = torch.cuda.is_available()

    # Random seed
    random.seed(time.time())
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    mkdir_p(args.out)

    args.n_gpus = len(args.gpus.split(','))
    state = {k: v for k, v in args._get_kwargs()}
    with open(os.path.join(args.out, 'args.json'), 'w', encoding='utf8') as f:
        json.dump(state, f)
        print('==> saved arguments')
    print(json.dumps(state, indent=4))
    set_seed(args)

    # Data
    print(f'==> Preparing IMDB')
    train_labeled_set, train_unlabeled_set, valid_set, test_set,\
    text_field, label_field = get_imdb('./data/aclImdb/')
    text_field.build_vocab(train_unlabeled_set, max_size=args.vocab_size,
                           vectors=GloVe(name='6B', dim=300, cache='./data/'))
    label_field.build_vocab(train_unlabeled_set)
    text_vocab, label_vocab = text_field.vocab, label_field.vocab
    print(f"Unique tokens in TEXT vocabulary: {len(text_vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(label_vocab)}")
    embedding_matrix = text_vocab.vectors
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

    config = Config(text_field, label_field, embedding=embedding_matrix)
    model = create_model(config, use_cuda=use_cuda)
    ema_model = create_model(config, use_cuda=use_cuda, ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

    ema_optimizer = WeightEMA(model, ema_model, args.lr, alpha=args.ema_decay)
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
        _, train_acc = validate(train_labeled_loader, ema_model, criterion, use_cuda, mode='Train Stats')
        val_loss, val_acc = validate(valid_loader, ema_model, criterion, use_cuda, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, ema_model, criterion, use_cuda, mode='Test Stats ')

        lr_scheduler.step(test_acc)

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
            'best_val_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.out)
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best val acc:')
    print(best_acc)

    print('Mean test acc:')
    print(np.mean(test_accs[-20:]))


def get_batch(iterator, loader):
    try:
        inputs, targets = next(iterator)
    except:
        iterator = iter(loader)
        inputs, targets = next(iterator)
    return iterator, inputs, targets

def get_max_length(tensors, pad_id):
    return max((tensor != pad_id).sum() for tensor in tensors)

def train(labeled_trainloader, unlabeled_trainloader, vocab, model, optimizer,
          ema_optimizer, criterion, epoch, use_cuda):
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
        pad_id = vocab.stoi[args.pad_token]
        max_len = max(get_max_length(inputs, pad_id) for inputs in [inputs_x, inputs_u, inputs_u2])
        inputs_x, inputs_u, inputs_u2 = inputs_x[:max_len], inputs_u[:max_len], inputs_u2[:max_len]
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
        logits_x, logits_u = model(input_a, input_b, l, mix=True)

        Lx, Lu, w = criterion(args.lambda_u, logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                              epoch + batch_idx / args.val_iteration, args.epochs)

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


def validate(valloader, model, criterion, use_cuda, mode):
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


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
