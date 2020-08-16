import torch
import torch.nn as nn
import torch.nn.functional as F
from util import interleave

class Config:

    def __init__(self, text, label, embedding=None):
        self.model_name = 'TextCNN'
        self.dropout = 0.5                                              # 随机失活
        self.num_classes = len(label.vocab)                             # 类别数
        self.n_vocab = len(text.vocab)                                  # 词表大小，在运行时赋值
        self.freeze = True
        self.embedding = embedding
        self.embed = embedding.size(1) \
            if embedding is not None else 300                           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


class MixTextCNN(nn.Module):
    def __init__(self, config):
        super(MixTextCNN, self).__init__()
        if config.embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding, freeze=config.freeze)
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
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # 256 * 3
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def forward(self, input_a, input_b=None, l=None, mix=False):
        input_a_emb = self.embedding(input_a)
        if mix:
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

