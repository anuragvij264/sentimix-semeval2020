import torch
from torch import optim
from nn_model.LSTM_attn import Classifier
from torch import nn
from config import bin_path, data_path
from torch.utils.data import DataLoader
from pipeline.trans import Dataset
from pipeline.dataloader import custom_collate
import os
from torch.nn.functional import cross_entropy
from scripts.save_and_load import save_ckp, load_ckp
import numpy as np

n_vocab = 100000
batch_size = 1000
n_embed = 300
n_hidden = 250
depth = 2
output_size = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Classifier(batch_size, n_hidden, depth, n_vocab, n_embed, output_size)
optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.01)

print_every = 100
step = 0
n_epochs = 20000  # validation loss increases from ~ epoch 3 or 4
clip = 5  # for gradient clip to prevent exploding gradient problem in LSTM/RNN


def load_embeddings(path, lang):
    weights = torch.load(open(os.path.join(path, "embeddings_{}.pth".format(lang)), "rb"))
    weights = torch.from_numpy(weights).double()
    # embeddings = nn.Embedding.from_pretrained(weights)
    return weights


hi_embeddings = load_embeddings(path=os.path.join(bin_path, "embeddings"), lang='hi')
en_embeddings = load_embeddings(path=os.path.join(bin_path, "embeddings"), lang='en')

d_set_val = Dataset(binPath=bin_path, data_path=data_path + 'data_val.json', hi_embeddings=hi_embeddings,
                    en_embeddings=en_embeddings)
test_loader = DataLoader(dataset=d_set_val, batch_size=1000, collate_fn=custom_collate, drop_last=True)


def test(model, ckp_path, validation_data_loader, optimizer):
    step = 0
    model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model, optimizer)
    model.eval()
    validation_loss = 0
    validation_accuracy = 0
    actual_labels = []
    predicted_labels = []
    tweet_ids = []
    for validation_inputs, validation_input_emoji, validation_input_profanity, validation_target, valid_ids in validation_data_loader:
        validation_inputs, validation_input_emoji, validation_input_profanity, validation_target = validation_inputs.to(
            device), validation_input_emoji.to(device), validation_input_profanity.to(
            device), validation_target.to(device)

        v_prediction = model(validation_inputs, validation_input_emoji, validation_input_profanity)
        v_loss = cross_entropy(v_prediction, validation_target.squeeze().long())
        validation_loss += v_loss.item()
        running_corrects_validation = (torch.max(v_prediction, 1)[1].view(validation_target.size()
                                                                          ).data == validation_target.data).float().sum()
        validation_accuracy = 100.0 * (running_corrects_validation.data / validation_inputs.size()[0])

        predicted_labels.append(torch.max(v_prediction, 1)[1].view(validation_target.size()).numpy())
        actual_labels.append(validation_target.data.numpy())
        tweet_ids.append(valid_ids)

        validation_loss += v_loss.item()
        validation_accuracy += validation_accuracy.item()
    result = {}
    result["predicted_labels"] = predicted_labels
    result["actual_labels"] = actual_labels
    result["tweet_ids"] = tweet_ids

    print("Step: {}".format(step),
          "Validation accuracy : {: .4f}".format(validation_accuracy.item() / 2))

    return result


ckp_path = "results/current_checkpoint.pt"

if __name__ == '__main__':
    result = test(net, ckp_path, test_loader, optimizer)
    for i in result.keys():
        result[i] = np.concatenate(result[i])

    f = open("results/answer.txt", "w")
    f.writelines("Uid,Sentiment\n")
    sentmap = {0: "positive", 1: "negative", 2: "neutral"}
    for i in range(len(result['actual_labels'])):
        f.writelines("{},{}\n".format(result['tweet_ids'][i], sentmap[result['predicted_labels'][i]]))
    f.close()
