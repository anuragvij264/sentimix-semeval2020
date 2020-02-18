import torch
import torch.nn as nn
from config import torch_emb_path
from torch.autograd import Variable
from torch.nn import functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):

    def __init__(self, batch_size, hidden_state_size, depth, vocab_size, embedding_length, output_size):
        super(Classifier, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_state_size = hidden_state_size
        self.depth = depth
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = self.init_embeddings(
            embeddings_path=torch_emb_path + "/embedding_weights.pth")

        # self.word_embeddings = self.load_embeddings(path=torch_emb_path)

        self.lstm = nn.LSTM(embedding_length, hidden_state_size, depth, dropout=0.3)
        self.label = nn.Linear(hidden_state_size + 2304 + 1, output_size)
        # self.sigmoid = nn.Sigmoid()

        # dont use softmax as cross entropy loss implements already
        self.softmax = nn.Softmax()

    def attention_net(self, lstm_output, final_state):
        final_state = final_state[:, -1:, :]
        hidden = final_state.reshape(final_state.shape[0], final_state.shape[1] * final_state.shape[2]).squeeze(0)
        attn_weights = torch.bmm(lstm_output[:, 0:-1, :], hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output[:, 0:-1, :].transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input_sentence, emoji_present, profanity, batch_size=None):
        input_sentence = self.word_embeddings(input_sentence.to(dtype=torch.long))
        input_sentence_embedding = input_sentence.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(self.depth, self.batch_size, self.hidden_state_size)).to(device)
        c_0 = Variable(torch.zeros(self.depth, self.batch_size, self.hidden_state_size)).to(device)

        output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence_embedding.float(), (h_0, c_0))
        output = output.permute(1, 0, 2)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_hidden_state)
        output_to_final_layer = torch.cat((attn_output, emoji_present.float(), profanity.float()), dim=1)
        linear_out = self.label(output_to_final_layer)
        return linear_out

    def lookup(self, id, embeddings):
        return embeddings[id]

    def init_embeddings(self, embeddings_path):
        weights = torch.load(open(embeddings_path, 'rb'))
        embeddings = nn.Embedding.from_pretrained(weights.to(device))
        return embeddings
