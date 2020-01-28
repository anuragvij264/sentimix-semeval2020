from deeptranslit import DeepTranslit
import json
import re
import pickle
import torch
from torch.utils import data
import os
import torch.nn as nn
import numpy as np
from config import data_path
from typing import Set, List

data_path = data_path + "data.json"

trans = DeepTranslit('hindi')
data_json = json.load(open(data_path, 'r'))

uids = data_json.keys()


def preprocess():
    tokens = dict()
    for uid in data_json.keys():
        temp_tok_list = []
        for token in data_json[uid]["text"]:
            if re.search(r'Hin', token):
                tok = token.split('\t')[0]
                try:
                    trans_tok = trans.transliterate(tok)
                    temp_tok_list.append(trans_tok)
                except:
                    temp_tok_list.append(tok)
            else:
                temp_tok_list.append(token.split('\t')[0])
    tokens[uid] = temp_tok_list
    return tokens


def preprocess2(data):
    pass


class Dataset(data.Dataset):

    def __init__(self, binPath, data_path, list_uids=None, list_data=None):
        self.word2ids = torch.load(os.path.join(binPath, "word2ids.pth"))
        self.uid2hin = torch.load(os.path.join(binPath, "uid2hin.pth"))
        # self.ids2word = torch.load(os.path.join(binPath, "ids2word.pth"))
        # TODO: Fix this mapping,
        self.ids2word = {j: i for i, j in self.word2ids.items()}
        self.word2id_en = torch.load(open(os.path.join(binPath, "embeddings", "word2id_en.pth"), "rb"))
        self.word2id_hi = torch.load(open(os.path.join(binPath, "embeddings", "word2id_hi.pth"), "rb"))
        self.data = json.load(open(data_path, 'rb'))
        self.list_uids = list(self.data.keys())

        self.hi_embeddings = self.load_embeddings(path=os.path.join(binPath, "embeddings"), lang='hi')
        self.en_embeddings = self.load_embeddings(path=os.path.join(binPath, "embeddings"), lang='en')

        self.uid2hi_index = torch.load(open(os.path.join(binPath, "uid2hi_idx.pth"), "rb"))
        self.embed_dim = 300

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        id = self.list_uids[index]
        X = self.tokenize(id, self.word2ids)
        idx_hi_list = set(self.uid2hi_index[id])

        X_ = self.tensorify_sentences(idx_hi_list, X)
        # print(X_)

        X_ = X_.permute(1, 0)
        y = self.data[id]["sent"]
        return X_, y

    def tokenize(self, uid, word2id):
        return [word2id[i.split('\t')[0]] for i in self.data[uid]['text']]

    def load_embeddings(self, path, lang):
        weights = torch.load(open(os.path.join(path, "embeddings_{}.pth".format(lang)), "rb"))
        weights = torch.from_numpy(weights).double()
        # embeddings = nn.Embedding.from_pretrained(weights)
        return weights

    def tensorify_sentences(self, idx_hi_list: Set[int], X: List[int]) -> torch.Tensor:
        array = np.zeros((self.embed_dim, len(X)))

        for i, _ in enumerate(X):
            word = self.ids2word[X[i]]

            if i in idx_hi_list:
                array[:, i] = self._lookup_embeddings(word, lang='hi')

            array[:, i] = self._lookup_embeddings(word, lang='en')

        resulting_tensor = torch.from_numpy(array)
        return resulting_tensor

    def _lookup_embeddings(self, word, lang):

        if lang == 'hi':
            word = self._transliterate(word)
            if word in self.word2id_hi:
                idx = self.word2id_hi[word]
                return self.hi_embeddings[idx]

            else:
                return np.zeros((1, self.embed_dim))
        else:
            if word in self.word2id_en:
                # print(word)
                idx = self.word2id_en[word]
                return self.en_embeddings[idx]

            else:
                # print(word)
                return np.zeros((1, self.embed_dim))

    def _transliterate(self, word):

        try:
            trans_word = trans.transliterate(word)
            return trans_word[0][0]
        # print(trans_word[0][0])
        except:

            # TODO: Code breaking here transliteration package breaks
            # - Temp fix to incorporate UNK token
            return "पंजाब"
