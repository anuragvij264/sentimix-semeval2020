from torch.utils.data import DataLoader
from pipeline.trans import Dataset
from keras.preprocessing import sequence
from config import bin_path, data_path
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_collate(batch):
    d = {"positive": 0, "negative": 1, "neutral": 2}
    sentence = [item[0] for item in batch]
    emojies = [item[1] for item in batch]
    profanities = [item[2] for item in batch]
    target = [item[3] for item in batch]

    length = [len(x) for x in sentence]
    data = sequence.pad_sequences(sentence, maxlen=max(length))
    data = torch.tensor(data, dtype=torch.double)
    data_emojies = torch.tensor(emojies, dtype=torch.double)
    data_profanities = torch.tensor(profanities, dtype=torch.double)

    target = list(map(lambda x: d[x], target))
    target = torch.tensor(target, dtype=torch.float32)

    _target = target.view((len(batch), 1))

    # target is of shape (batch_size,1)
    return [data, data_emojies, data_profanities, target]


def load_embeddings(path, lang):
    weights = torch.load(open(os.path.join(path, "embeddings_{}.pth".format(lang)), "rb"))
    weights = torch.from_numpy(weights).double()
    # embeddings = nn.Embedding.from_pretrained(weights)
    return weights


hi_embeddings = load_embeddings(path=os.path.join(bin_path, "embeddings"), lang='hi')
en_embeddings = load_embeddings(path=os.path.join(bin_path, "embeddings"), lang='en')

d_set = Dataset(binPath=bin_path, data_path=data_path + 'data.json', hi_embeddings=hi_embeddings,
                en_embeddings=en_embeddings)
loader = DataLoader(dataset=d_set, batch_size=12, collate_fn=custom_collate)

#
# example = []
# from tqdm import tqdm
#
# for idx, (batch_ndx, sample) in tqdm(enumerate(loader)):
#     print(batch_ndx.shape)
#     print(sample)
# #     print(sample.size())
#     print(sample.dtype)
#
