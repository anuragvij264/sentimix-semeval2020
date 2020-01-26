from torch.utils.data import DataLoader
from pipeline.trans import Dataset
from config import bin_path, data_path
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_collate(batch):

    print(batch)
    # print(type(batch))

    batch = [torch.Tensor(t[0]).to(device) for t in batch]

    batch = torch.nn.utils.rnn.pad_sequence(batch)
    return batch


d_set = Dataset(binPath=bin_path, data_path=data_path + 'data.json')
loader = DataLoader(dataset=d_set, batch_size=12,collate_fn=custom_collate)



for idx, (batch_ndx, sample) in enumerate(loader):
    print(sample)
    break