import torch
import re
import json
dat = json.load(open('data/data.json','rb'))

wrds_set = set()
for i in dat:
    wrds = set([k.split('\t')[0] for k in dat[i]["text"]])
    wrds_set.update(wrds)

word2ids = {j:i for i,j in enumerate(wrds_set)}
ids2word = {i:j for i,j in enumerate(wrds_set)}
torch.save(word2ids,open("word2ids.pth","wb"))
torch.save(word2ids,open("ids2word.pth","wb"))


def uid2hin(word2ids,data):
    uid2hi = dict()
    for i in data:
       uid2hi[i]=[word2ids[k.split('\t')[0]]  for k in data[i]["text"] if re.search(r'Hin',k)]
    return uid2hi
