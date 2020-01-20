from deeptranslit import DeepTranslit
import json
import re
import pickle
import torch
from torch.utils import data
import os

data_path = "/Users/avij1/Desktop/imp_shit/sec/data/data.json"


trans = DeepTranslit('hindi')
data_json = json.load(open(data_path,'r'))

uids = data_json.keys()

def preprocess():
 tokens = dict()
 for uid in data_json.keys():
  temp_tok_list = []
  for token in data_json[uid]["text"]:
      if re.search(r'Hin',token): #token.split('\t')[1]=="Hin": ## or use regex
        tok = token.split('\t')[0]
        # print(tok)
        try:
         trans_tok = trans.transliterate(tok)
         temp_tok_list.append(trans_tok)
        except:
         # print(tok)
         temp_tok_list.append(tok)
      else: temp_tok_list.append(token.split('\t')[0])
 tokens[uid] = temp_tok_list
 print(temp_tok_list)
 return tokens

def preprocess2(data):
    pass

    # for uid in data:
    #     = data[uid]["text"]
    #




class Dataset(data.Dataset):

    def __init__(self,binPath,data_path,list_uids=None):
        self.word2ids = torch.load(os.path.join(binPath,"word2ids.pth"))
        self.uid2hin = torch.load(os.path.join(binPath,"uid2hin.pth"))
        self.ids2word = torch.load(os.path.join(binPath,"ids2word.pth"))
        self.data = json.load(open(data_path,'rb'))
        self.list_uids = list(self.data.keys())


    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):

        id = self.list_uids[index]
        X =[i.split('\t')[0] for i in self.data[id]["text"]]

        # X = torch.load('data/' + id + '.pt')
        y = self.data[id]["sent"]
        return X,y


def load_embeddings():


# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#
#     binPath = "/Users/avij1/Desktop/imp_shit/sec/bin"
#     data_path = "/Users/avij1/Desktop/imp_shit/sec/data/data.json"
#     ds = Dataset(binPath,data_path)
#
#     for i,j in DataLoader(dataset=ds,batch_size=50):
#         print(i,j)
#         break

