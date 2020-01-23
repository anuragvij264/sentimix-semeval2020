import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable
import os


class Classifier(nn.Module):

    def __init__(self,batch_size,hidden_state_size,vocab_size,embedding_length,output_size):
        super(Classifier,self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_state_size = hidden_state_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = self.load_embeddings(path="../bin/embeddings",lang='en')


        self.lstm = nn.LSTM(embedding_length,hidden_state_size)

        self.label = nn.Linear(hidden_state_size,output_size)

    def forward(self,input_sentence,batch_size=None):
        input = self.word_embeddings(input_sentence)

        input = input.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_state_size)).to(device)
        c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_state_size)).to(device)


        out,(final_hidden_state,final_cell_state)=self.lstm(input,(h_0,c_0))


        #
        # print(out)

        # h0 = Variable(torch.zeros(1,self.batch_size,self.hidden_state_size)).to(device)
        # c0  = Variable(torch.zeros(1,self.batch_size,self.output_size)).to(device)
        # output,(_,_)= self.lstm(input,(h0,c0))
        #
        # logits = self.label(output)

        return input

    def load_embeddings(self,path,lang):

        weights = torch.load(open(os.path.join(path,"embeddings_{}.pth".format(lang)),"rb"))
        weights = torch.from_numpy(weights).double()
        # id2word = torch.load(open(os.path.join(path, "id2word_{}.pth".format(lang))))
        # word2id = torch.load(open(os.path.join(path, "word2id_{}.pth".format(lang))))
        embeddings = nn.Embedding.from_pretrained(weights)
        return embeddings


    def lookup(self,word,embeddings,word2id):
        id = word2id[word]

        #TODO: ** Add logic for OOV words ** . vocab here refers to the one made by MUSE

        return embeddings[id]


if __name__=='__main__':

    params = {
    "batch_size" : 2,
    "hidden_state_size" :10,
    "vocab_size":100000,
    "embedding_length":300,
    "output_size": 3
    }
    cls = Classifier(**params)

    print(cls.forward(torch.LongTensor(
        [[12,123,87,980],[23,34,76,98]]
    )))