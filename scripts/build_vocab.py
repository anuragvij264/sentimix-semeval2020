import os
import torch
import pickle
import numpy as np



Code_dir = "/Users/avij1/Desktop/imp_shit/sec/"
bin_path = Code_dir + "bin/"

# loading transliterated words

trans_words = pickle.load(open('/Users/avij1/Desktop/imp_shit/sec/scripts/file_log.pkl','rb')) ## transliterated pairs
en_set = pickle.load(open('/Users/avij1/Desktop/imp_shit/sec/scripts/en.pkl','rb')) # unique english words


def load_embeddings(path, lang):
    weights = torch.load(open(os.path.join(path, "embeddings_{}.pth".format(lang)), "rb"))
    weights = torch.from_numpy(weights).double()

    # unk = weights.shape[0]

    weights = torch.cat((weights,torch.zeros(1,300).double()))

    print(weights[100000])

    # embeddings = nn.Embedding.from_pretrained(weights)
    return weights

def build_global_vocab(hi_words : set = set(trans_words.keys()),en_words : set = en_set):
    """
    take input en and hi words in the corpus and
    return global word2idx and idx2word
    :return:
    """
    _ = [hi_words.add(i) for i in en_words]
    all_words = hi_words

    word2idx_global = {w:idx for idx,w in enumerate(all_words)}
    word2idx_global['<UNK>'] = len(word2idx_global)


    return word2idx_global


# load ids
word2id_hi = torch.load(open(os.path.join(bin_path, "embeddings", "word2id_hi.pth"), "rb"))
word2id_hi['<UNK>'] = len(word2id_hi)
word2id_en = torch.load(open(os.path.join(bin_path, "embeddings", "word2id_en.pth"), "rb"))
word2id_en['<UNK>'] = len(word2id_en)
hi_embeddings = load_embeddings(path=os.path.join("/Users/avij1/Desktop/imp_shit/sec/bin", "embeddings"), lang='hi')
en_embeddings = load_embeddings(path=os.path.join("/Users/avij1/Desktop/imp_shit/sec/bin", "embeddings"), lang='en')


import numpy as np


# print(np.max(list(word2id_hi.values())))

# print(len(word2id_hi))

"""
[
    [       ],
    [       ],
    [       ],
]

X --> vocab keys 
Y --> embeddings 

store indexes too 
"""

"""
word2idx --> global word mapping
hi_embeddings 
lookup
 
 """
vocab = build_global_vocab()

torch.save(vocab,'vocab.pth')

embed_dim = 300

arr = np.zeros((len(vocab),300))

for i in vocab:

    if i in trans_words: # only those hien words which have been transliterated
        if i in word2id_hi:
            word_hi_idx = word2id_hi[i]

            arr_idx = vocab[i]
            arr[arr_idx,:] = hi_embeddings[word_hi_idx]

        else: # transliterations not present in the hi dictionary # UNK
            word_hi_idx = word2id_hi['<UNK>']
            arr_idx = vocab[i]
            arr[arr_idx,:] = hi_embeddings[word_hi_idx]

    elif i in word2id_en:
        # fetch for the english ones

        word_en_idx = word2id_en[i]
        arr_idx = vocab[i]
        arr[arr_idx,:] = en_embeddings[word_en_idx]
    else:
        word_en_idx = word2id_en["<UNK>"]
        # arr[word_hi_idx]

        # print(i)
        arr_idx = vocab[i]
        arr[arr_idx,:] = en_embeddings[word_en_idx]


embedding_weights = torch.from_numpy(arr)
torch.save(embedding_weights,'embedding_weights.pth')

        # <UNK>
# for i in trans_words:
#     w = trans_words[i]
#     idx = word2idx[w]
#     if w in word2id_hi:
#        arr[idx,] = lookup(w,hi_embeddings)
#     else:





