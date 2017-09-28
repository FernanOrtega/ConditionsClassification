import numpy as np
from keras import Input

from keras.engine import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, AveragePooling2D, Conv2D, GlobalMaxPooling2D, Conv1D, \
    GlobalMaxPooling1D, MaxPooling2D, Flatten, Masking, AveragePooling1D, Lambda, MaxPooling1D
from keras.layers import concatenate
from keras.preprocessing import sequence

maxlen = 500


class ModelContainer(object):
    def inputs_context_emb_layer(self, X, wv_model):
        vocab = wv_model.wv.vocab
        return np.array([np.array([np.array([vocab.get(t).index for t in row[0]]),
                                   np.array([vocab.get(t).index for t in row[1]]),
                                   np.array([vocab.get(t).index for t in row[2]]),
                                   np.array([vocab.get(t).index for t in row[3]])])
                         for row in X])

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


# Added mask -> Terrific results!!
class ModelContainer2(ModelContainer):
    def create_model(self, X, wv_model):
        X_preprocessed = super(ModelContainer2, self).inputs_nocontext_emb_layer(X, wv_model)
        X_preprocessed = sequence.pad_sequences(X_preprocessed, maxlen=maxlen, padding='post', truncating='post',
                                                value=-1)

        embedding_layer = wv_model.wv.get_embedding_layer()
        # embedding_layer.mask_zero=True
        sequence_input = Input(shape=(maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)

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


# Bad effectiveness results: acc=0.38
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


# Emb layer with mask, sequence 3 conv layers
class ModelContainer7(ModelContainer):
    def create_model(self, X, wv_model):
        X_preprocessed = super(ModelContainer7, self).inputs_nocontext_emb_layer(X, wv_model)
        X_preprocessed = sequence.pad_sequences(X_preprocessed, maxlen=maxlen, padding='post', truncating='post',
                                                value=-1)

        embedding_layer = wv_model.wv.get_embedding_layer()
        # embedding_layer.mask_zero=True
        sequence_input = Input(shape=(maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)

        x = (Conv1D(filters=128,
                    kernel_size=4,
                    activation='relu'))(embedded_sequences)
        x = (AveragePooling1D(pool_size=8))(x)
        x = (Dropout(0.2))(x)
        x = (Conv1D(filters=256,
                    kernel_size=2,
                    activation='relu'))(x)
        x = (Dropout(0.5))(x)
        x = (Conv1D(filters=32,
                    kernel_size=8,
                    activation='relu'))(x)
        x = (GlobalMaxPooling1D())(x)
        x = (Dense(16))(x)
        x = (Dropout(0.2))(x)
        preds = (Dense(2, activation='sigmoid'))(x)
        model = Model(sequence_input, preds, name='Model7')

        return model, X_preprocessed


# Emb layer with mask, multiple conv branches
class ModelContainer8(ModelContainer):
    def create_model(self, X, wv_model):
        X_preprocessed = super(ModelContainer8, self).inputs_nocontext_emb_layer(X, wv_model)
        X_preprocessed = sequence.pad_sequences(X_preprocessed, maxlen=maxlen, padding='post', truncating='post',
                                                value=-1)

        embedding_layer = wv_model.wv.get_embedding_layer()
        # embedding_layer.mask_zero=True
        sequence_input = Input(shape=(maxlen,), dtype='int32')
        mask = Masking(mask_value=-1)(sequence_input)
        embedded_sequences = embedding_layer(mask)

        x_1_c1 = Conv1D(filters=64,
                        kernel_size=2,
                        activation='relu')(embedded_sequences)
        x_1_c1 = MaxPooling1D()(x_1_c1)
        x_1_c1 = GlobalMaxPooling1D()(x_1_c1)

        x_1_c2 = Conv1D(filters=64,
                        kernel_size=4,
                        activation='relu')(embedded_sequences)
        x_1_c2 = MaxPooling1D()(x_1_c2)
        x_1_c2 = GlobalMaxPooling1D()(x_1_c2)

        x_1_c3 = Conv1D(filters=64,
                        kernel_size=8,
                        activation='relu')(embedded_sequences)
        x_1_c3 = MaxPooling1D()(x_1_c3)
        x_1_c3 = GlobalMaxPooling1D()(x_1_c3)

        x_1_c4 = Conv1D(filters=64,
                        kernel_size=16,
                        activation='relu')(embedded_sequences)
        x_1_c4 = MaxPooling1D()(x_1_c4)
        x_1_c4 = GlobalMaxPooling1D()(x_1_c4)

        x = concatenate([x_1_c1, x_1_c2, x_1_c3, x_1_c4])
        x = Dense(64)(x)
        x = Dropout(0.3)(x)
        x = Dense(16)(x)
        preds = (Dense(2, activation='sigmoid'))(x)
        model = Model(sequence_input, preds, name='Model8_multiple_conv')

        return model, X_preprocessed


# Taking contexts into account (4 convolution branches and one concatenation step), embedding layer, padding and mask
class ModelContainer9(ModelContainer):
    def create_model(self, X, wv_model):
        X_preprocessed = super(ModelContainer9, self).inputs_context_emb_layer(X, wv_model)
        X_preprocessed = np.array([sequence.pad_sequences(subX_prep, maxlen=maxlen, padding='post',
                                                          truncating='post', value=-1) for subX_prep in X_preprocessed])

        input = Input(shape=(4, maxlen), dtype='int32')

        # Left Context branch
        input_L = Lambda(lambda x: x[:, 0], output_shape=(maxlen,))(input)
        embedding_layer_L = wv_model.wv.get_embedding_layer()
        mask_L = Masking(mask_value=-1)(input_L)
        emb_seq_L = embedding_layer_L(mask_L)
        x_L = Conv1D(filters=128,
                     kernel_size=4,
                     activation='relu')(emb_seq_L)
        x_L = GlobalMaxPooling1D()(x_L)

        # Connective branch
        input_N = Lambda(lambda x: x[:, 1], output_shape=(maxlen,))(input)
        embedding_layer_N = wv_model.wv.get_embedding_layer()
        mask_N = Masking(mask_value=-1)(input_N)
        emb_seq_N = embedding_layer_N(mask_N)
        x_N = Conv1D(filters=128,
                     kernel_size=4,
                     activation='relu')(emb_seq_N)
        x_N = GlobalMaxPooling1D()(x_N)

        # Condition branch
        input_C = Lambda(lambda x: x[:, 2], output_shape=(maxlen,))(input)
        embedding_layer_C = wv_model.wv.get_embedding_layer()
        mask_C = Masking(mask_value=-1)(input_C)
        emb_seq_C = embedding_layer_C(mask_C)
        x_C = Conv1D(filters=128,
                     kernel_size=4,
                     activation='relu')(emb_seq_C)
        x_C = GlobalMaxPooling1D()(x_C)

        # Right context branch
        input_R = Lambda(lambda x: x[:, 3], output_shape=(maxlen,))(input)
        embedding_layer_R = wv_model.wv.get_embedding_layer()
        mask_R = Masking(mask_value=-1)(input_R)
        emb_seq_R = embedding_layer_R(mask_R)
        x_R = Conv1D(filters=128,
                     kernel_size=4,
                     activation='relu')(emb_seq_R)
        x_R = GlobalMaxPooling1D()(x_R)

        x = concatenate([x_L, x_N, x_C, x_R])
        x = Dense(64)(x)
        x = Dropout(0.3)(x)
        x = Dense(16)(x)
        preds = (Dense(2, activation='sigmoid'))(x)
        model = Model(input, preds, name='Model9_contexts')

        return model, X_preprocessed
