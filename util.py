import errno
import os
import shutil

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn
from torch.utils.data import Dataset

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class AverageMeter:
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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


def interleave_offsets(batch, nu):
    """
    Args:
        batch: batch_size
        nu: n 个 batch 减一

    Returns:

    """
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    """
    Args:
        xy: n 个 batch 的 list
        batch: batch_size

    Returns:

    例子：
    batch_size: 128
    len(xy) = 6, 即有 6 个 batch
    那么经过 interleave_offsets 处理得到：
      labeled:  [21, 21, 21, 21, 22, 22]
    unlabeled: [[21, 21, 21, 21, 22, 22]
                [21, 21, 21, 21, 22, 22]
                [21, 21, 21, 21, 22, 22]
                [21, 21, 21, 21, 22, 22]
                [21, 21, 21, 21, 22, 22]]
    如果直接将以上每一行作为一个 batch 输入到模型中，那么情况变成第一个 batch 全是标注数据，后面的数据全部都是无标注数据
    所以下面函数将对角线上分别与第一行同一列进行调换以使得每一个 batch 都存在一部分标注数据，每一个 batch 的数据分布也变得较为一致
    """
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def savefig(fname, dpi=150):
    plt.savefig(fname, dpi=dpi)


def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]


class Logger:
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


class LoggerMonitor:
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    _, pred = torch.max(output, dim=1)
    correct = pred.float().eq(target).float()
    return correct.mean()


class SemiLoss:
    def __call__(self, lambda_u, outputs_x, targets_x, outputs_u, targets_u, cur_epoch, total_epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        # 标注数据损失
        # Lx = -torch.mean(torch.sum(torch.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lx = nn.BCEWithLogitsLoss()(outputs_x, targets_x)
        # 无标注数据损失
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, lambda_u * linear_rampup(cur_epoch, total_epoch)  # 线性增大 lambda_u


class WeightEMA:
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

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


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def truncated_padded(token_ids, pad_id, max_length):
    if len(token_ids) < max_length:
        token_ids.extend([pad_id] * (max_length - len(token_ids)))
    else:
        token_ids = token_ids[:max_length]
    return token_ids


class MyIMDB(Dataset):

    def __init__(self, imdb, text_vocab, label_vocab, max_length=512, unlabeled=False):
        super(MyIMDB, self).__init__()
        self.imdb = imdb
        self.text_vocab = text_vocab
        self.label_vovab = label_vocab
        self.unlabeled = unlabeled
        self.max_length = max_length

    def __getitem__(self, idx):
        if self.unlabeled:
            example_a = self.imdb[2*idx]
            example_b = self.imdb[2*idx+1]
            input_a = [self.text_vocab.stoi[token] for token in example_a.text]
            input_b = [self.text_vocab.stoi[token] for token in example_b.text]
            input_a = truncated_padded(input_a, self.text_vocab.stoi['<pad>'], self.max_length)
            input_b = truncated_padded(input_b, self.text_vocab.stoi['<pad>'], self.max_length)
            return torch.tensor(input_a, dtype=torch.long), torch.tensor(input_b, dtype=torch.long)
        else:
            example = self.imdb[idx]
            input = [self.text_vocab.stoi[token] for token in example.text]
            input = truncated_padded(input, self.text_vocab.stoi['<pad>'], self.max_length)
            target = self.label_vovab.stoi[example.label]
            return torch.tensor(input, dtype=torch.long), torch.tensor(target, dtype=torch.float)

    def __len__(self):
        return len(self.imdb) if not self.unlabeled else len(self.imdb) // 2