import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import time
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from keras.utils import np_utils
from keras.metrics import categorical_accuracy, mae

import matplotlib.pyplot as plt

from model_creator import ModelContainer1
from model_creator import ModelContainer2

np.random.seed(123)

# Turning off interactive mode of PyPlot
plt.ioff()


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


def get_model_containers():
    models = list()
    models.append(ModelContainer1())
    models.append(ModelContainer2())

    return models


def load_wv_model(wv_model_path):
    start = time.time()
    we_model = Word2Vec.load(wv_model_path)
    end = time.time()

    print('Time loading w2v model: ', (end - start))

    return we_model


def precision_recall_curve(model_name, k, y_test, y_pred):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    average_precision = average_precision_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve (model=' + str(model_name) + ', k=' + str(k) + ': AUC={0:0.2f}'.format(
        average_precision))

    plt.savefig('results/img/p-r-curve-'+model_name+'-'+str(k)+'.png')


def cm_measures(y_test, y_pred):
    return precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), \
           roc_auc_score(y_test, y_pred)


if __name__ == '__main__':
    wv_model_path = 'models/w2v-ciaoreviews'
    dataset_path = 'C:/datasets/es/dataset'
    labels_path = 'C:/datasets/es/dataset.labels'
    results_path = 'results/benchmark-results.csv'

    X, Y = load_dataset(dataset_path, labels_path)
    wv_model = load_wv_model(wv_model_path)

    performance_results = list()
    performance_results.append(['Model', 'K', 'keras_loss', 'keras_acc', 'keras_mae', 'keras_cat_acc', 'tp', 'fn', 'fp', 'tn',
                                'P', 'R', 'F1', 'AUC-ROC', 'learning_time', 'test_time'])

    folds = StratifiedKFold(n_splits=4, random_state=7, shuffle=False)
    splits = [(train_index, test_index) for train_index, test_index in folds.split(X, Y)]

    # Testing
    Y = np_utils.to_categorical(Y, 2)
    models_containers = get_model_containers()
    for model_container in models_containers:
        model, X_preprocessed = model_container.create_model(X, wv_model)
        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', mae, categorical_accuracy])
        print(model.summary())
        init_weights = model.get_weights()
        k = 1
        for train_index, test_index in splits:
            X_train, X_test = X_preprocessed[train_index], X_preprocessed[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            # Fit model on training data
            start = time.time()
            model.fit(X_train, Y_train, batch_size=100, epochs=10, verbose=1)
            end = time.time()
            learn_time = end - start
            score = model.evaluate(X_test, Y_test, verbose=1)

            start = time.time()
            y_pred = model.predict(X_test)
            end = time.time()
            predict_time = end - start
            y_pred = np.array([np.argmax(i) for i in y_pred])

            y_test = np.array([np.argmax(i) for i in Y_test])
            cf = confusion_matrix(y_test, y_pred)
            p, r, f1, auc_roc = cm_measures(y_test, y_pred)

            performance_results.append(
                [model.name, k, score[0], score[1], score[2], score[3], cf[1][1], cf[1][0], cf[0][1],
                 cf[0][0], p, r, f1, auc_roc, learn_time, predict_time])

            precision_recall_curve(model.name, k, y_test, y_pred)

            model.save('models/'+model.name+'k'+str(k))
            # Reset learnt weights to execute new experiment
            model.set_weights(init_weights)
            k += 1


    df = pd.DataFrame(data=performance_results[1:], columns=performance_results[0])
    df.set_index(performance_results[0][0])
    df.to_csv(results_path)
