from torch.utils.data import DataLoader
from pipeline.trans import Dataset
from config import bin_path, data_path

d_set = Dataset(binPath=bin_path, data_path=data_path + 'data.json')
loader = DataLoader(dataset=d_set, batch_size=12)

for idx, (batch_ndx, sample) in enumerate(loader):
    print(sample)
