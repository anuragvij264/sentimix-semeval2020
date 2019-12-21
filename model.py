import torch
import torch.nn as nn
from torch.autograd import Variable
import bcolz
import numpy as np
en_vec_path = "/Users/avij1/Desktop/imp_shit/fasttext/MUSE/dumped/debug/w1c3c03p4w/vectors-en.txt"
hi_vec_path = "/Users/avij1/Desktop/imp_shit/fasttext/MUSE/dumped/debug/w1c3c03p4w/vectors-hi.txt"

vectors = bcolz.carray(np.zeros(1),rootdir = en_vec_path,mode='w')


# with open("{}",'rb') as file:
#     for l in file:
#         line = l.decode().split()
#
#         word = line[0]
#         words.append(word)
#
#         word2idx[word] = idx
#         idx +=1
#
#         vect = np.array(line[1:]).astype(np.float)
#         vectors.append(vect)
