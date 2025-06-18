from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
import time
import os

def delete_stopwords(words):
    _words = []
    for i in range(len(words)):
        _words.append(remove_stopwords(words[i]).split(" "))

    return _words

def cut_words_key(words, key):
    _words = np.asarray(words)
    # 短补0，长截断
    for i in range(_words.shape[0]):
        if len(_words[i]) < key:
            index = len(_words[i])
            for j in range(key - len(_words[i])):
                _words[i].append(0)
    __words = []
    for i in range(_words.shape[0]):
        __words.append(_words[i][:key])
    return __words

def charToEmbed(words, embed):
    embed_list = []
    word_embed = []
    zero_embed = [float(0) for i in range(50)]
    for i in range(len(words)):
        for j  in range(len(words[0])):
            if words[i][j] in embed.keys():
                words[i][j] = embed[words[i][j]]
            else:
                words[i][j] = zero_embed

    return words

def get_glove(path):
    vocab, embedding = [], []
    with open(path, 'rt', encoding = 'utf8') as f:
        full_content = f.read().strip().split('\n')

    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(var) for var in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embedding.append(i_embeddings)

    vocab_npa = vocab
    embs_npa = embedding

    dict_words_embed = {}
    for i in range(len(vocab_npa)):
        dict_words_embed[vocab_npa[i]] = embs_npa[i]

    return dict_words_embed

def get_fold_indexed(train_index, label, n_split = 5):
    skf = StratifiedKFold(n_splits = n_split, shuffle = True, random_state = 42)
    fold_index = []
    train_index = np.array(train_index)
    for fold, (fold_train_idx, fold_val_idx) in enumerate(skf.split(train_index, label)):
        actual_train_index = train_index[fold_train_idx]
        actual_val_index = train_index[fold_val_idx]

        fold_index.append({
            "train": actual_train_index.tolist(),
            "val": actual_val_index.tolist()
        })
    return fold_index
