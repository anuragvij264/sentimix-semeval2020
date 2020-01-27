import torch
from torch import optim
from nn_model.LSTM_attn import Classifier
from torch import nn
from config import bin_path, data_path
from torch.utils.data import DataLoader
from pipeline.trans import Dataset
from pipeline.dataloader import custom_collate
import numpy as np

n_vocab = 100000
batch_size = 12
n_embed = 300
n_hidden = 10
output_size = 2

net = Classifier(batch_size, n_hidden, n_vocab, n_embed, output_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

print_every = 100
step = 0
n_epochs = 4  # validation loss increases from ~ epoch 3 or 4
clip = 5  # for gradient clip to prevent exploding gradient problem in LSTM/RNN
device = 'cuda' if torch.cuda.is_available else 'cpu'

d_set_train = Dataset(binPath=bin_path, data_path=data_path + 'data.json')
train_loader = DataLoader(dataset=d_set_train, batch_size=12, collate_fn=custom_collate)

d_set_val = Dataset(binPath=bin_path, data_path=data_path + 'data.json')
valid_loader = DataLoader(dataset=d_set_val, batch_size=12, collate_fn=custom_collate)

for epoch in range(n_epochs):

    for inputs, labels in train_loader:
        step += 1
        inputs, labels = inputs.to(device), labels.to(device)
        h = tuple([each.data for each in h])

        net.zero_grad()
        output_train = net(inputs)
        loss = criterion(output_train.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), clip)
        optimizer.step()

        if (step % print_every) == 0:
            ######################
            ##### VALIDATION #####
            ######################
            net.eval()
            valid_losses = []

            for v_inputs, v_labels in valid_loader:
                v_inputs, v_labels = inputs.to(device), labels.to(device)

                v_output = net(v_inputs)
                v_loss = criterion(v_output.squeeze(), v_labels.float())
                valid_losses.append(v_loss.item())

            print("Epoch: {}/{}".format((epoch + 1), n_epochs),
                  "Step: {}".format(step),
                  "Training Loss: {:.4f}".format(loss.item()),
                  "Validation Loss: {:.4f}".format(np.mean(valid_losses)))
            net.train()
