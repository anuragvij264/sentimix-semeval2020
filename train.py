import torch
from torch import optim
from nn_model.LSTM_Bidirectional_attn import Classifier
from torch import nn
from config import bin_path, data_path
from torch.utils.data import DataLoader
from pipeline.trans import Dataset
from pipeline.dataloader import custom_collate
import os
from torch.nn.functional import cross_entropy
from scripts.save_and_load import save_ckp, load_ckp

n_vocab = 100000
batch_size = 1000
n_embed = 300
n_hidden = 250
output_size = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Classifier(batch_size, n_hidden, n_vocab, n_embed, output_size)

optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.1)

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

d_set_train = Dataset(binPath=bin_path, data_path=data_path + 'data.json', hi_embeddings=hi_embeddings,
                      en_embeddings=en_embeddings)
train_loader = DataLoader(dataset=d_set_train, batch_size=1000, collate_fn=custom_collate, drop_last=True)

d_set_val = Dataset(binPath=bin_path, data_path=data_path + 'data_val.json', hi_embeddings=hi_embeddings,
                    en_embeddings=en_embeddings)
valid_loader = DataLoader(dataset=d_set_val, batch_size=1000, collate_fn=custom_collate, drop_last=True)


def train(model, training_data_loader, validation_data_loader):
    from tqdm import tqdm
    previous_validation_accuracy = 0
    step = 0
    model.train()
    loss_total_epoch = 0
    accuracy_total_epoch = 0
    for epoch in range(n_epochs):
        for inputs, input_emoji, input_profanity, target, train_id in tqdm(training_data_loader):
            inputs = inputs.float()
            inputs, input_emoji, input_profanity, target = inputs.to(device), input_emoji.to(
                device), input_profanity.to(device), target.to(device)
            model.zero_grad()
            prediction = model(inputs, input_emoji, input_profanity)
            # prediction is of size (batch_size,C)
            # target is of size (batch_size)
            loss = cross_entropy(prediction, target.squeeze().long())

            loss_total_epoch += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()

            running_corrects = (torch.max(prediction, 1)[1].view(target.size()
                                                                 ).data == target.data).float().sum()
            accuracy = 100.0 * (running_corrects.data / inputs.size()[0])

            accuracy_total_epoch += accuracy.item()
            step += 1
        print("training Step : {} ".format(step))
        if (epoch % 5) == 0:
            model.eval()
            validation_loss = 0
            validation_accuracy = 0

            for validation_inputs, validation_input_emoji, validation_input_profanity, validation_target, val_id in validation_data_loader:
                validation_inputs, validation_input_emoji, validation_input_profanity, validation_target = validation_inputs.to(
                    device), validation_input_emoji.to(device), validation_input_profanity.to(
                    device), validation_target.to(device)

                v_prediction = model(validation_inputs, validation_input_emoji, validation_input_profanity)
                v_loss = cross_entropy(v_prediction, validation_target.squeeze().long())
                validation_loss += v_loss.item()

                running_corrects_validation = (torch.max(v_prediction, 1)[1].view(validation_target.size()
                                                                                  ).data == validation_target.data).float().sum()
                validation_accuracy = 100.0 * (running_corrects_validation.data / validation_inputs.size()[0])

                validation_loss += v_loss.item()
                validation_accuracy += validation_accuracy.item()
                print(val_id,running_corrects_validation.data)

            print("Epoch: {}/{}".format((epoch + 1), n_epochs),
                  "Step: {}".format(step),
                  "Training Loss: {:.4f}".format(loss.item()),
                  # "Validation Loss: {:.4f}".format(validation_loss),
                  "Validation accuracy : {: .4f}".format(validation_accuracy.item() / 2),
                  "Training accuracy:  :{: .4f}".format(accuracy.item()),

                  )
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': validation_accuracy.item() / 2,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if validation_accuracy.item() / 2 > previous_validation_accuracy:
                save_ckp(checkpoint, True, 'dummy/current_checkpoint.pt', 'dummy/best_checkpoint.pt')
            else:
                save_ckp(checkpoint, False, 'dummy/current_checkpoint.pt', 'dummy/best_checkpoint.pt')

            previous_validation_accuracy = validation_accuracy.item() / 2


if __name__ == '__main__':
    train(net, train_loader, valid_loader)
