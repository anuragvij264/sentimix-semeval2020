import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable

class Classifier(nn.Module):

    def __init__(self,batch_size,hidden_state_size,vocab_size,embedding_length,output_size,weights):
        super(Classifier,self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_state_size = hidden_state_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding(vocab_size,embedding_length)

        self.word_embeddings.weight = nn.Parameter(weights,requires_grad= False)

        self.lstm = nn.LSTM(embedding_length,hidden_state_size)

        self.label = nn.linear(hidden_state_size,output_size)

    def forward(self,input_sentence,batch_size=None):
        input = self.word_embeddings(input_sentence)
        h0 = Variable(torch.zeros(1,self.batch_size,self.hidden_state_size)).to(device)
        c0  = Variable(torch.zeros(1,self.batch_size,self.output_size)).to(device)
        output,(_,_)= self.lstm(input,(h0,c0))

        logits = self.label(output)

        return logits

# if __name__=='__main__':
#     cls = Classifier()
