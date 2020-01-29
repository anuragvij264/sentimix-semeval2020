from torch.utils.data import DataLoader
from pipeline.trans import Dataset
from keras.preprocessing import sequence
from config import bin_path, data_path
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_collate(batch):

    d = {"positive":0,"negative":1,"neutral":2}

    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    length = [len(x) for x in data]
    data = sequence.pad_sequences(data, maxlen=max(length))
    data = torch.tensor(data, dtype=torch.double)

    target = list(map(lambda x: d[x] ,target))
    target = torch.tensor(target, dtype=torch.float32)

    _target = target.view((len(batch),1))

    # target is of shape (batch_size,1)
    return [data, _target]


d_set = Dataset(binPath=bin_path, data_path=data_path + 'data.json')
loader = DataLoader(dataset=d_set, batch_size=12, collate_fn=custom_collate)

#
# example = []
#
# for idx, (batch_ndx, sample) in enumerate(loader):
#     print(batch_ndx.shape)
#     print(sample)
#     print(sample.size())
#     print(sample.dtype)
