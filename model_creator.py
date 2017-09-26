import numpy as np
from keras import Input

from keras.engine import InputLayer, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, AveragePooling2D, Conv2D, GlobalMaxPooling2D, Conv1D, AveragePooling1D, \
    GlobalMaxPooling1D
from keras.preprocessing import sequence

maxlen = 500


class ModelContainer(object):
    def inputs_nocontext_array_emb(self, X, wv_model):
        return np.array([np.array([wv_model[t] for t in np.concatenate([row[1], row[2]])]) for row in X])

    def inputs_nocontext_emb_layer(self, X, wv_model):
        return np.array([np.array([wv_model.wv.vocab.get(t).index for t in np.concatenate([row[1], row[2]])]) for row in X])


class ModelContainer1(ModelContainer):
    def create_model(self, X, wv_model):
        X_preprocessed = super(ModelContainer1, self).inputs_nocontext_array_emb(X, wv_model)
        X_preprocessed = sequence.pad_sequences(X_preprocessed, maxlen=maxlen, padding='post', truncating='post')
        X_preprocessed = X_preprocessed.reshape(X_preprocessed.shape[0], 1, X_preprocessed.shape[1], X_preprocessed.shape[2])

        model = Sequential(name='Model1')
        model.add(Conv2D(filters=128,
                         kernel_size=(4, 4),
                         activation='relu',
                         input_shape=(1, X_preprocessed.shape[2], X_preprocessed.shape[3])))
        # model.add(AveragePooling2D(pool_size=(8, 8)))
        # model.add(Dropout(0.5))
        # model.add(Conv2D(filters=256,
        #                  kernel_size=(2, 2),
        #                  activation='relu'))
        model.add(GlobalMaxPooling2D())
        model.add(Dense(16))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='sigmoid'))

        return model, X_preprocessed


class ModelContainer2(ModelContainer):
    def create_model(self, X, wv_model):
        X_preprocessed = super(ModelContainer2, self).inputs_nocontext_emb_layer(X, wv_model)
        X_preprocessed = sequence.pad_sequences(X_preprocessed, maxlen=maxlen, padding='post', truncating='post', value=-1)

        embedding_layer = wv_model.wv.get_embedding_layer()
        sequence_input = Input(shape=(maxlen,), dtype='int32')

        embedded_sequences = embedding_layer(sequence_input)

        x = (Conv1D(filters=128,
                         kernel_size=4,
                         activation='relu'))(embedded_sequences)
        # x = (AveragePooling1D(pool_size=8))(x)
        # x = (Dropout(0.5))(x)
        # x = (Conv1D(filters=256,
        #                  kernel_size=2,
        #                  activation='relu'))(x)
        x = (GlobalMaxPooling1D())(x)
        x = (Dense(16))(x)
        x = (Dropout(0.2))(x)
        preds = (Dense(2, activation='sigmoid'))(x)
        model = Model(sequence_input, preds, name='Model2')

        return model, X_preprocessed
