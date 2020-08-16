from torchtext import datasets
from torchtext import data
from torchtext.vocab import GloVe
import torch.optim as opt
import torch.nn as nn
import torch

import os
from tqdm import tqdm
import argparse

from textcnn import MixTextCNN, Config
from util import accuracy

parser = argparse.ArgumentParser(description='PyTorch TextCNN Training')

parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='train batch-size')
parser.add_argument('--vocab-size', default=50000, type=int, metavar='N',
                    help='vocabulary size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--num_labeled', default=500, type=int,
                    help='the number of labeled examples')

parser.add_argument('--device', default='cpu', type=str,
                    help='training device')
parser.add_argument('--gpus', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

def train(model, train_iterator, criterion, optimizer, device):
    preds = []
    labels = []
    running_avg_loss = .0
    model.train()
    for i, batch in tqdm(enumerate(train_iterator)):
        x = batch.text.t().to(device)
        pred = model(x)
        y = batch.label.to(device)
        loss = criterion(pred[:, 1], y)

        running_avg_loss += (loss.item() - running_avg_loss) / (i + 1)
        preds.append(pred)
        labels.append(y)

        print(f'running average training loss: {running_avg_loss:.3f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    acc = accuracy(preds, labels)
    print(f'training accuracy is: {acc:.3f} \n')


def valid(model, valid_iterator, device, mode='dev'):
    print('evaluation starting ...')
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(valid_iterator):
            x = batch.text.t().to(device)
            pred = model(x)
            preds.append(pred)
            labels.append(batch.label.to(device))
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    acc = accuracy(preds, labels)
    print(f'{mode} accuracy is: {acc:.3f} \n')


def main(args):
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    text = data.Field()
    label = data.LabelField(dtype=torch.float)

    trainset, testset = datasets.IMDB.splits(text, label, path='./data/aclImdb/')
    trainset, validset = trainset.split(split_ratio=args.num_labeled / 25000)

    text.build_vocab(testset, max_size=args.vocab_size,
                     vectors=GloVe(name='6B', dim=300, cache='./data/'))
    label.build_vocab(validset)
    config = Config(text, label, embedding=text.vocab.vectors)

    print(f"Unique tokens in TEXT vocabulary: {len(text.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(label.vocab)}")


    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (trainset, validset, testset), batch_size=args.batch_size
    )

    model = MixTextCNN(config)
    model.to(args.device)

    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        print(f'Epoch [{epoch+1}/{args.epochs}] starting ...\n')
        train(model, train_iterator, criterion, optimizer, args.device)
        valid(model, valid_iterator, args.device)

    valid(model, test_iterator, args.device, mode='test')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)








