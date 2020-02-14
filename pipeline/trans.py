import json
import torch
from torch.utils import data
import os
import torch.nn as nn
import numpy as np
from config import data_path, bin_path
from typing import Set, List


class Dataset(data.Dataset):

    def __init__(self, binPath, data_path, hi_embeddings, en_embeddings, list_uids=None, list_data=None):
        self.word2ids = torch.load(os.path.join(binPath, "word2ids.pth"))
        # TODO: Fix this mapping,
        self.ids2word = {j: i for i, j in self.word2ids.items()}
        self.word2id_en = torch.load(open(os.path.join(binPath, "embeddings", "word2id_en.pth"), "rb"))
        self.word2id_hi = torch.load(open(os.path.join(binPath, "embeddings", "word2id_hi.pth"), "rb"))
        self.data = json.load(open(data_path, 'rb'))
        self.list_uids = list(self.data.keys())
        self.vocab = torch.load(open(bin_path + '/vocab.pth', 'rb'))

        self.hi_embeddings = hi_embeddings
        self.en_embeddings = en_embeddings

        self.uid2hi_index = torch.load(open(os.path.join(binPath, "embeddings", "uid2hi_idx.pth"), "rb"))
        self.embed_dim = 300

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        id = self.list_uids[index]
        X = self.tokenize(id, self.vocab)
        X_emoji = self.data[id]["emoticons"]
        X_profanity = [1 * (len(self.data[id]["profanities"]) > 0)]
        y = self.data[id]["sent"]
        return X, X_emoji, X_profanity, y, id

    def tokenize(self, uid, word2id):

        tokens_list = []
        for i in self.data[uid]['text']:
            word = i.split('\t')[0]
            if word in word2id:
                tokens_list.append(word2id[word])
            else:
                tokens_list.append(word2id['<UNK>'])

        return tokens_list
