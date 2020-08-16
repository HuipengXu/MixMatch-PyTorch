import random
import io
import os
import shutil
from tqdm import tqdm

from augment import eda

import torch
from torchtext import data

random.seed(1)


def mkdir(directory):
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(directory, 'pos'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'neg'), exist_ok=True)


def augmentation(data_dir, num_labeled, split_ratio):
    aug_labeled_dir = os.path.join(data_dir, 'aug_train_labeled')
    aug_unlabeled_dir = os.path.join(data_dir, 'aug_train_unlabeled')
    valid_dir = os.path.join(data_dir, 'valid')
    orig_dir = os.path.join(data_dir, 'train')
    # 清除原有文件
    if os.path.exists(aug_labeled_dir):
        shutil.rmtree(aug_labeled_dir)
    if os.path.exists(aug_unlabeled_dir):
        shutil.rmtree(aug_unlabeled_dir)
    if os.path.exists(valid_dir):
        shutil.rmtree(valid_dir)

    mkdir(aug_labeled_dir)
    mkdir(aug_unlabeled_dir)
    mkdir(valid_dir)
    for label in ['pos', 'neg']:
        count = 0
        path = os.path.join(orig_dir, label)
        for i, file in tqdm(enumerate(os.listdir(path))):
            if i >= int(split_ratio * len(os.listdir(path))):
                shutil.copyfile(os.path.join(path, file), os.path.join(valid_dir, label, file))
                continue
            ranking = file.split('.')[0].split('_')[-1]
            if i == num_labeled // 2: count = 0
            num_aug = 1 if i < num_labeled // 2 else 2
            aug_dir = aug_labeled_dir if i < num_labeled // 2 else aug_unlabeled_dir
            with open(os.path.join(path, file), 'r', encoding='utf8') as orig:
                text = orig.read()
                aug_texts = eda(text, num_aug=num_aug)
                for t in aug_texts:
                    with open(os.path.join(aug_dir, label, str(count) + '_' + ranking + '.txt'),
                              'w', encoding='utf8') as aug:
                        aug.write(t)
                    count += 1


class IMDB(data.Dataset):
    def __init__(self, path, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        for label in ['pos', 'neg']:
            cur_path = os.path.join(path, label)
            files = [file for file in os.listdir(cur_path) if len(file.split('_')) <= 2]
            files = sorted(files, key=lambda x: int(x.split('_')[0]))
            for fname in files:
                with io.open(os.path.join(cur_path, fname), 'r', encoding="utf-8") as f:
                    text = f.readline()
                examples.append(data.Example.fromlist([text, label], fields))

        super(IMDB, self).__init__(examples, fields, **kwargs)


def get_imdb(data_dir):
    text_field = data.Field()
    label_field = data.LabelField(dtype=torch.float)
    train_labeled_dataset = IMDB(data_dir + 'aug_train_labeled', text_field, label_field)
    train_unlabeled_dataset = IMDB(data_dir + 'aug_train_unlabeled', text_field, label_field)
    valid_dataset = IMDB(data_dir + 'valid', text_field, label_field)
    test_dataset = IMDB(data_dir + 'test', text_field, label_field)
    print(f'Total {len(train_labeled_dataset)} labeled examples')
    print(f'Total {len(train_unlabeled_dataset)} unlabeled examples')
    print(f'Total {len(valid_dataset)} valid examples')
    print(f'Total {len(test_dataset)} test examples')
    return train_labeled_dataset, train_unlabeled_dataset, \
           valid_dataset, test_dataset, text_field, label_field


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('augment data')
    parser.add_argument('--data_dir', type=str, default='./data/aclImdb/')
    parser.add_argument('--num_labeled_examples', type=int, default=500)
    parser.add_argument('--split_ratio', type=float, default=0.7)
    args = parser.parse_args()

    augmentation(args.data_dir, args.num_labeled_examples, args.split_ratio)
