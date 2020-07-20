from __future__ import print_function, absolute_import
import torch
__all__ = ['accuracy']

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""

    _, pred = torch.max(output, dim=1)
    correct = pred.float().eq(target).float()
    return correct.mean()

if __name__ == '__main__':
    a = torch.arange(18).reshape(9, 2)
    b = torch.tensor([1,1,0,0,0,0,1,1,1])
    print(accuracy(a, b))
