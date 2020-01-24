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

def uid2hi_idx(data: dict) -> dict:
    uid2hi_index = dict()

    for i in data:
        index_dict = {n: m for m, n in enumerate(data[i]["text"])}
        uid2hi_index[i] = [index_dict[w] for w in index_dict if re.search(r'Hin',w)]

    return uid2hi_index
#
# if __name__=='__main__':
#
#      d = uid2hi_idx(data = dat)
#      torch.save(d,open("uid2hi_idx.pth","wb"))