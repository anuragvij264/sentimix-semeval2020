import os
import io
import numpy as np
import torch

emb_path = '/Users/avij1/Desktop/imp_shit/fasttext/MUSE/dumped/debug/375u9teo6w/'

def load_vec(emb_path, nmax=100000):

    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

def dump_embds(embedding,id2word,word2id,path,lang):

    torch.save(embedding,open(os.path.join(path,"embeddings_{}.pth".format(lang)),"wb"))
    torch.save(id2word, open(os.path.join(path,"id2word_{}.pth".format(lang)), "wb"))
    torch.save(word2id, open(os.path.join(path,"word2id_{}.pth".format(lang)), "wb"))

def main():
    embeddings_hi,id2word_hi,word2id_hi = load_vec(emb_path=os.path.join(emb_path,'vectors-hi.txt'))
    embeddings_en,id2word_en,word2id_en = load_vec(emb_path=os.path.join(emb_path,'vectors-en.txt'))

    dump_embds(embeddings_hi,id2word_hi,word2id_hi,path='../bin/embeddings',lang='hi')
    dump_embds(embeddings_en,id2word_en,word2id_en,path='../bin/embeddings',lang='en')

if __name__=='__main__':
    main()