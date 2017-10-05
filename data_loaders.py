import time, warnings

import numpy as np
from gensim.models import Word2Vec

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def load_dataset(dataset_path, labels_path):
    start = time.time()
    X = [eval(line) for line in open(dataset_path, encoding='utf-8')]
    end = time.time()
    print('Time loading dataset: ', (end - start))

    X = np.array(X)
    start = time.time()
    y = np.array([line for line in open(labels_path, encoding='utf-8')])
    end = time.time()
    print('Time loading labels: ', (end - start))

    return X, y


def load_wv_model(wv_model_path):
    start = time.time()
    we_model = Word2Vec.load(wv_model_path)
    end = time.time()

    print('Time loading w2v model: ', (end - start))

    return we_model