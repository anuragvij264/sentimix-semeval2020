import pandas as pd
import json
import re
from gensim import utils
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath

import logging

logging.basicConfig(level=logging.INFO)

data = json.load(open('data.json', 'r'))


def _get_words(tweet):
    """
    input: takes a list of words (with lang tags)
    output: just words

    """
    return [w.split('\t')[0] for w in tweet]


class DataIter(object):

    # @staticmethod
    # def _get_words(df):
    #     uids = [i for i in df['uid2'].values]
    #     for i in uids:
    #         [df['uid2']==i]['meta']

    def __iter__(self):
        for uid in data.keys():
            yield _get_words(data[uid]["text"])


if __name__ == '__main__':
    data_ = DataIter()

    model = FastText(sg=1, hs=1, min_n=4, max_n=6)
    model.build_vocab(sentences=DataIter())
    total_examples = model.corpus_count
    model.train(sentences=DataIter(), total_examples=total_examples, epochs=10)

    temp_file = datapath("model")

    # print(model.wv.vocab)
    # fname = get_tmpfile("f.model")
    model.save(temp_file)
