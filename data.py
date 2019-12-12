import pandas as pd


df = pd.read_csv('train.csv',index = False)




class DataBatch():

    @staticmethod
    def _get_words(df):
        print(df.head())


    def __iter__(self):
        pass
