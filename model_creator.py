import numpy as np
from keras import Input

from keras.engine import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, AveragePooling2D, Conv2D, GlobalMaxPooling2D, Conv1D, \
    GlobalMaxPooling1D, MaxPooling2D, Flatten
from keras.layers import concatenate
from keras.preprocessing import sequence

maxlen = 500


class ModelContainer(object):
    def inputs_nocontext_array_emb(self, X, wv_model):
        return np.array([np.array([wv_model[t] for t in np.concatenate([row[1], row[2]])]) for row in X])

    def inputs_nocontext_emb_layer(self, X, wv_model):
        return np.array(
            [np.array([wv_model.wv.vocab.get(t).index for t in np.concatenate([row[1], row[2]])]) for row in X])


class ModelContainerBase(ModelContainer):
    def create_model(self, X, wv_model):
        # Specific preprocessing of the dataset
        X_preprocessed = X

        # NN architecture
        model = Sequential(name='ModelBase')
        # model.add(...
        #
        # or...
        #
        # x = Conv1D(...
        # model = Model(name='ModelBase', input, x)

        return model, X_preprocessed


class ModelContainer1(ModelContainer):
    def create_model(self, X, wv_model):
        X_preprocessed = super(ModelContainer1, self).inputs_nocontext_array_emb(X, wv_model)
        X_preprocessed = sequence.pad_sequences(X_preprocessed, maxlen=maxlen, padding='post', truncating='post')
        X_preprocessed = X_preprocessed.reshape(X_preprocessed.shape[0], 1, X_preprocessed.shape[1],
                                                X_preprocessed.shape[2])

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
        X_preprocessed = sequence.pad_sequences(X_preprocessed, maxlen=maxlen, padding='post', truncating='post',
                                                value=-1)

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


class ModelContainer3(ModelContainer):
    def create_model(self, X, wv_model):
        X_preprocessed = super(ModelContainer3, self).inputs_nocontext_array_emb(X, wv_model)
        X_preprocessed = sequence.pad_sequences(X_preprocessed, maxlen=maxlen, padding='post', truncating='post')
        X_preprocessed = X_preprocessed.reshape(X_preprocessed.shape[0], 1, X_preprocessed.shape[1],
                                                X_preprocessed.shape[2])

        model = Sequential(name='Model3')
        model.add(Conv2D(filters=128,
                         kernel_size=(4, 4),
                         activation='relu',
                         input_shape=(1, X_preprocessed.shape[2], X_preprocessed.shape[3])))
        model.add(AveragePooling2D(pool_size=(8, 8)))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=256,
                         kernel_size=(2, 2),
                         activation='relu'))
        model.add(GlobalMaxPooling2D())
        model.add(Dense(16))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='sigmoid'))

        return model, X_preprocessed


class ModelContainer4(ModelContainer):
    def create_model(self, X, wv_model):
        X_preprocessed = super(ModelContainer4, self).inputs_nocontext_array_emb(X, wv_model)
        X_preprocessed = sequence.pad_sequences(X_preprocessed, maxlen=maxlen, padding='post', truncating='post')
        X_preprocessed = X_preprocessed.reshape(X_preprocessed.shape[0], 1, X_preprocessed.shape[1],
                                                X_preprocessed.shape[2])

        model = Sequential(name='Model4')
        model.add(Conv2D(filters=64,
                         kernel_size=(4, 4),
                         activation='relu',
                         input_shape=(1, X_preprocessed.shape[2], X_preprocessed.shape[3])))
        model.add(AveragePooling2D(pool_size=(4, 4)))
        model.add(Dropout(0.2))
        model.add(Conv2D(filters=128,
                         kernel_size=(2, 2),
                         activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=16,
                         kernel_size=(8, 8),
                         activation='relu'))
        model.add(GlobalMaxPooling2D())
        model.add(Dense(16))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='sigmoid'))

        return model, X_preprocessed


class ModelContainer5(ModelContainer):
    def create_model(self, X, wv_model):
        X_preprocessed = super(ModelContainer5, self).inputs_nocontext_array_emb(X, wv_model)
        X_preprocessed = sequence.pad_sequences(X_preprocessed, maxlen=maxlen, padding='post', truncating='post')
        X_preprocessed = X_preprocessed.reshape(X_preprocessed.shape[0], 1, X_preprocessed.shape[1],
                                                X_preprocessed.shape[2])

        input = Input(shape=(1, X_preprocessed.shape[2], X_preprocessed.shape[3]))

        x_1_c1 = Conv2D(filters=128,
                         kernel_size=(2, int(X_preprocessed.shape[3] / 2)),
                         activation='relu')(input)
        x_1_c1 = MaxPooling2D()(x_1_c1)
        x_1_c1 = GlobalMaxPooling2D()(x_1_c1)

        x_1_c2 = Conv2D(filters=64,
                         kernel_size=(4, int(X_preprocessed.shape[3] / 4)),
                         activation='relu')(input)
        x_1_c2 = MaxPooling2D()(x_1_c2)
        x_1_c2 = GlobalMaxPooling2D()(x_1_c2)

        x_1_c3 = Conv2D(filters=32,
                         kernel_size=(8, int(X_preprocessed.shape[3] / 8)),
                         activation='relu')(input)
        x_1_c3 = MaxPooling2D()(x_1_c3)
        x_1_c3 = GlobalMaxPooling2D()(x_1_c3)

        x_1_c4 = Conv2D(filters=16,
                         kernel_size=(16, int(X_preprocessed.shape[3] / 16)),
                         activation='relu')(input)
        x_1_c4 = MaxPooling2D()(x_1_c4)
        x_1_c4 = GlobalMaxPooling2D()(x_1_c4)

        x = concatenate([x_1_c1, x_1_c2, x_1_c3, x_1_c4])
        x = Dense(256)(x)
        x = Dropout(0.4)(x)
        x = Dense(32)(x)
        x = Dropout(0.25)(x)
        preds = (Dense(2, activation='sigmoid'))(x)
        model = Model(input, preds, name='Model5_multiple_conv')

        return model, X_preprocessed


class ModelContainer6(ModelContainer):
    def create_model(self, X, wv_model):
        X_preprocessed = super(ModelContainer6, self).inputs_nocontext_array_emb(X, wv_model)
        X_preprocessed = sequence.pad_sequences(X_preprocessed, maxlen=maxlen, padding='post', truncating='post')
        X_preprocessed = X_preprocessed.reshape(X_preprocessed.shape[0], 1, X_preprocessed.shape[1],
                                                X_preprocessed.shape[2])

        input = Input(shape=(1, X_preprocessed.shape[2], X_preprocessed.shape[3]))

        x_1_c1 = Conv2D(filters=64,
                         kernel_size=(2, 2),
                         activation='relu')(input)
        x_1_c1 = MaxPooling2D()(x_1_c1)
        x_1_c1 = GlobalMaxPooling2D()(x_1_c1)

        x_1_c2 = Conv2D(filters=64,
                         kernel_size=(4, 4),
                         activation='relu')(input)
        x_1_c2 = MaxPooling2D()(x_1_c2)
        x_1_c2 = GlobalMaxPooling2D()(x_1_c2)

        x_1_c3 = Conv2D(filters=64,
                         kernel_size=(8, 8),
                         activation='relu')(input)
        x_1_c3 = MaxPooling2D()(x_1_c3)
        x_1_c3 = GlobalMaxPooling2D()(x_1_c3)

        x_1_c4 = Conv2D(filters=64,
                         kernel_size=(16, 16),
                         activation='relu')(input)
        x_1_c4 = MaxPooling2D()(x_1_c4)
        x_1_c4 = GlobalMaxPooling2D()(x_1_c4)

        x = concatenate([x_1_c1, x_1_c2, x_1_c3, x_1_c4])
        x = Dense(64)(x)
        x = Dropout(0.3)(x)
        x = Dense(16)(x)
        preds = (Dense(2, activation='sigmoid'))(x)
        model = Model(input, preds, name='Model6_multiple_conv')

        return model, X_preprocessed
