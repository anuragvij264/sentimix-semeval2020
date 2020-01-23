from deeptranslit import DeepTranslit
import json
import re
import pickle
import torch
from torch.utils import data
import os
import torch.nn as nn
from config import data_path

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
        self.ids2word = torch.load(os.path.join(binPath, "ids2word.pth"))
        self.word2id_en = torch.load(open(os.path.join(binPath, "embeddings", "word2id_en.pth"), "rb"))
        self.word2id_hi = torch.load(open(os.path.join(binPath, "embeddings", "word2id_hi.pth"), "rb"))
        self.data = json.load(open(data_path, 'rb'))
        self.list_uids = list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        id = self.list_uids[index]
        X = self.tokenize(id, self.word2ids)
        y = self.data[id]["sent"]
        return X, y

    def tokenize(self, uid, word2id):
        return [word2id[i.split('\t')[0]] for i in self.data[uid]['text']]
