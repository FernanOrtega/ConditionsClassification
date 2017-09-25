import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim, os, time


class WordEmbeddings(object):
    def __init__(self, parentdir, modelpath, workers=1, re_train=False, min_count=1,
                 vector_size=100):
        self.parentdir = parentdir
        if not re_train and os.path.isfile(modelpath):
            self.model = gensim.models.Word2Vec.load(modelpath)
        elif os.path.exists(parentdir):
            start = time.time()
            self.model = gensim.models.Word2Vec(self, min_count=min_count, workers=workers
                                                , size=vector_size)
            end = time.time()
            print('Learning time: ', (end - start))
            self.model.save(modelpath)

    # This method could raise an error if no sentences were found.
    def __iter__(self):
        for dirpath, dnames, fnames in os.walk(self.parentdir):
            for fname in fnames:
                if fname.endswith("sentences.tok"):
                    for line in open(os.path.join(dirpath, fname), encoding='utf-8'):
                        yield eval(line)

    # For now, we don't deal with the problem of unknown words
    def get_embeddings(self, word):
        return self.model[word]

# we = WordEmbeddings(parentdir='C:/datasets/es/headset', modelpath='models/w2vtest', re_train=True)
#
# for line in we:
#     print(line[0])
