import pandas as pd

train, test = ("train_conll.txt", "trial_conll.txt")

df = pd.read_csv(train, sep='\t', names=["meta", "uid", "sentiment"])

# with open(train,'rb') as file:
#     text = file.readlines()
# df.columns = ["meta","uid","sentiment"]

print(df.head())


# print(int(df['uid'][0]))

def get_uid(x):
    try:
        a = int(x)
        return a
    except:
        return None


def get_lang(x):
    try:
        a = int(x)
        return None
    except:
        return str(x)


df['lang'] = df['uid'].apply(lambda x: get_lang(x))
df['uid2'] = df['uid'].apply(lambda x: get_uid(x))

df.to_csv('train.csv')
# print(df.head())
