from torch.utils.data import DataLoader
from pipeline.trans import Dataset

dset = Dataset(binPath="/Users/avij1/Desktop/imp_shit/sec/bin",data_path="/Users/avij1/Desktop/imp_shit/sec/data/data.json")
loader = DataLoader(dataset=dset,batch_size=12)

for idx,(batch_ndx,sample) in enumerate(loader):
    print(sample)
