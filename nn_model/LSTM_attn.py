import torch
import torch.nn as nn
from config import torch_emb_path
from torch.autograd import Variable
from torch.nn import functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):

    def __init__(self, batch_size, hidden_state_size, vocab_size, embedding_length, output_size):
        super(Classifier, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_state_size = hidden_state_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = self.load_embeddings(path=torch_emb_path, lang='en')

        self.lstm = nn.LSTM(embedding_length, hidden_state_size)
        self.label = nn.Linear(hidden_state_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input_sentence, batch_size=None):
        input_sentence_embedding = self.word_embeddings(input_sentence)
        input_sentence_embedding = input_sentence_embedding.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_state_size)).to(device)
        c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_state_size)).to(device)

        output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence_embedding, (h_0, c_0))
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)
        return self.softmax(logits)

    def load_embeddings(self, path, lang):
        weights = torch.load(open(os.path.join(path, "embeddings_{}.pth".format(lang)), "rb"))
        weights = torch.from_numpy(weights).double()
        embeddings = nn.Embedding.from_pretrained(weights)
        return embeddings

    def lookup(self, word, embeddings, word2id):
        id = word2id[word]

        # TODO: ** Add logic for OOV words ** . vocab here refers to the one made by MUSE

        return embeddings[id]


if __name__ == '__main__':
    params = {
        "batch_size": 2,
        "hidden_state_size": 10,
        "vocab_size": 100000,
        "embedding_length": 300,
        "output_size": 3
    }
    cls = Classifier(**params)
    input = torch.LongTensor([[12, 123, 87, 980], [23, 34, 76, 98]]).to(device)
    print(cls.forward(input))
