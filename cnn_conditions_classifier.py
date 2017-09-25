import numpy as np
from sklearn.model_selection import train_test_split

import time
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec

np.random.seed(123)

from os.path import isfile

from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, AveragePooling2D, MaxPooling2D, Conv2D, GlobalMaxPooling2D
from keras.utils import np_utils
from keras.metrics import categorical_accuracy, binary_accuracy, mae


def create_model(model_path):
    model = Sequential()
    model.add(Conv2D(filters=128,
                     kernel_size=(4, 4),
                     activation='relu',
                     input_shape=(1, X_train.shape[2], X_train.shape[3])))
    model.add(AveragePooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=256,
                     kernel_size=(2, 2),
                     activation='relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(16))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', mae, binary_accuracy])

    # Fit model on training data
    start = time.time()
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
    end = time.time()
    print('Learning time: ', (end - start))

    model.save(model_path)

    return model


model_w2v = 'models/w2v-ciaoreviews'
dataset_path = 'C:/datasets/es/dataset'
labels_path = 'C:/datasets/es/dataset.labels'
model_path = 'models/cnn_conditions_model.h5'

# Load Word2Vec model
start = time.time()
we_model = Word2Vec.load(model_w2v)
end = time.time()

print('Time loading w2v model: ', (end - start))
# Load data (simplified)
maxlen = 0
start = time.time()
X = []

for line in open(dataset_path, encoding='utf-8'):
    e_line = eval(line)
    conn_cond = np.concatenate([e_line[1], e_line[2]])
    if conn_cond.size > maxlen:
        maxlen = conn_cond.size
    v_line = np.array([we_model[t] for t in conn_cond])
    X.append(v_line)
end = time.time()
print('Time loading dataset: ', (end - start))

X = np.array(X)
start = time.time()
y = np.array([line for line in open(labels_path, encoding='utf-8')])
end = time.time()
print('Time loading labels: ', (end - start))

y = y.astype('float32')
# Padding
X = sequence.pad_sequences(X, maxlen=maxlen)

X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
# Split dataset into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
Y_train = np_utils.to_categorical(y_train, 2)

Y_test = np_utils.to_categorical(y_test, 2)
print('X_train shape:', X_train.shape)

print('X_test shape:', X_test.shape)
# Building model
if isfile(model_path):
    model = load_model(model_path)
else:
    model = create_model(model_path)

print(model.summary())

# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=1)

print('Score: ', score)
