# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
import io
import os
import shutil
from tqdm import tqdm

random.seed(1)

import torch
from torch.utils.data import Dataset
from torchtext import data
import numpy as np

# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']

# cleaning up text
import re


def get_only_chars(line):
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

# for the first time you use wordnet
# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    # sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))

    # ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))

    # rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))

    # rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    # append the original sentence
    # augmented_sentences.append(sentence)

    return augmented_sentences

def mkdir(directory):
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(directory, 'pos'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'neg'), exist_ok=True)

def augmentation(root='../data/aclImdb/', num_labeled=250, split_ratio=0.7):
    aug_labeled_dir = os.path.join(root, 'aug_train_labeled')
    aug_unlabeled_dir = os.path.join(root, 'aug_train_unlabeled')
    valid_dir = os.path.join(root, 'valid')
    orig_dir = os.path.join(root, 'train')
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

def get_imdb(root='../data/aclImdb/'):
    text_field = data.Field()
    label_field = data.LabelField(dtype=torch.float)
    train_labeled_dataset = IMDB(root + 'aug_train_labeled', text_field, label_field)
    train_unlabeled_dataset = IMDB(root + 'aug_train_unlabeled', text_field, label_field)
    valid_dataset = IMDB(root+'valid', text_field, label_field)
    test_dataset = IMDB(root + 'test', text_field, label_field)
    print(f'Total {len(train_labeled_dataset)} labeled examples')
    print(f'Total {len(train_unlabeled_dataset)} unlabeled examples')
    print(f'Total {len(valid_dataset)} valid examples')
    print(f'Total {len(test_dataset)} test examples')
    return train_labeled_dataset, train_unlabeled_dataset, \
           valid_dataset, test_dataset, text_field, label_field


if __name__ == '__main__':
    augmentation(num_labeled=500)
