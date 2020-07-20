import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import interleave

class Config(object):

    """配置参数"""
    def __init__(self, text, label, embedding=None):
        self.model_name = 'TextCNN'
        self.dropout = 0.5                                              # 随机失活
        self.num_classes = len(label.vocab)                             # 类别数
        self.n_vocab = len(text.vocab)                                  # 词表大小，在运行时赋值
        self.num_epochs = 5                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率
        self.embedding = embedding
        self.embed = embedding.size(1) \
            if embedding is not None else 300                           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MixTextCNN(nn.Module):
    def __init__(self, config):
        super(MixTextCNN, self).__init__()
        if config.embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) # shape (bs, hidden, seq_len)
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # shape (bs, hidden)
        return x

    def _forward(self, x):
        out = x
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # 256 * 3
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def forward(self, input_a, input_b=None, l=None, training=False):
        input_a_emb = self.embedding(input_a)
        if training:
            batch_size = input_a.size(0)
            input_b_emb = self.embedding(input_b)
            # 13、14行合成一步了
            mixed_input_emb = l * input_a_emb + (1 - l) * input_b_emb

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input_emb, batch_size//3))
            mixed_input = interleave(mixed_input, batch_size)

            logits = [self._forward(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(self._forward(input))

            # put interleaved samples back
            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)
            return logits_x, logits_u
        else:
            return self._forward(input_a_emb)

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        if config.embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) # shape (bs, hidden, seq_len)
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # shape (bs, hidden)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # 256 * 3
        out = self.dropout(out)
        out = self.fc(out)
        return out

def accuracy(preds, label):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    preds = np.where(preds>=0.5, 1, 0)
    acc = accuracy_score(label, preds)
    return acc

######################### baseline #########################
if __name__ == '__main__':
    from torchtext import datasets
    from torchtext import data
    import torch.optim as opt
    from sklearn.metrics import accuracy_score
    import numpy as np
    from tqdm import tqdm

    text = data.Field()
    label = data.LabelField(dtype=torch.float)

    trainset, testset = datasets.IMDB.splits(text, label, path='../data/aclImdb/')
    trainset, validset = trainset.split(split_ratio=0.70)

    max_vocab_size = 25000
    text.build_vocab(trainset, max_size=max_vocab_size)
    label.build_vocab(trainset)
    config = Config(text, label)

    print(f"Unique tokens in TEXT vocabulary: {len(text.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(label.vocab)}")


    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (trainset, validset, testset), batch_size=config.batch_size
    )

    model = TextCNN(config)
    model.to(config.device)

    optimizer = opt.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config.num_epochs):
        print(f'Epoch [{epoch+1}/{config.num_epochs}] starting ...')
        preds = []
        labels = []
        running_avg_loss = .0
        model.train()
        for i, batch in tqdm(enumerate(train_iterator)):
            x = batch.text.t().to(config.device)
            pred = model(x)[:, 1]
            y = batch.label.to(config.device)
            loss = criterion(pred, y)

            running_avg_loss += (loss.item() - running_avg_loss) / (i + 1)
            preds.append(pred)
            labels.append(y.cpu().numpy())

            print(f'running average training loss: {running_avg_loss:.3f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = torch.cat(preds, dim=0)
        labels = np.concatenate(labels, axis=0)
        acc = accuracy(preds, labels)
        print(f'training accuracy is: {acc:.3f}')

        print('evaluation starting ...')
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(valid_iterator):
                x = batch.text.t().to(config.device)
                pred = model(x)[:, 1]
                preds.append(pred)
                labels.append(batch.label.numpy())
        preds = torch.cat(preds, dim=0)
        labels = np.concatenate(labels, axis=0)
        acc = accuracy(preds, labels)
        print(f'valid accuracy is: {acc:.3f}')

    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(test_iterator):
            x = batch.text.t().to(config.device)
            pred = model(x)[:, 1]
            preds.append(pred)
            labels.append(batch.label.numpy())
    preds = torch.cat(preds, dim=0)
    labels = np.concatenate(labels, axis=0)
    acc = accuracy(preds, labels)
    print(f'test accuracy is: {acc:.3f}')








