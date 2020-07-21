'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os

import torch

__all__ = ['mkdir_p', 'AverageMeter', 'interleave']


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
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