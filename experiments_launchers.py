import gc, os, time

import numpy as np
from keras.callbacks import EarlyStopping, Callback
from keras.metrics import mae, binary_accuracy
from keras.utils import np_utils
from six import iteritems
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from evaluation import precision_recall_curve, cm_measures

np.random.seed(123)


def experiments_with_K_fold(X, Y, wv_model, n_k_splits, batch_size, epochs, results_path, history_path, img_folder_path,
                            models_folder_path, models_containers):
    performance_results = list()
    folds = StratifiedKFold(n_splits=n_k_splits, random_state=7, shuffle=False)
    splits = [(train_index, test_index) for train_index, test_index in folds.split(X, Y)]
    Y = np_utils.to_categorical(Y, 2)
    for model_container in models_containers:
        print(type(model_container).__name__)
        model, X_preprocessed = model_container.create_model(X, wv_model)
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', mae, binary_accuracy])
        print(model.summary())
        init_weights = model.get_weights()
        k = 1
        for train_index, test_index in splits:
            X_train, X_test = X_preprocessed[train_index], X_preprocessed[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            # Fit model on training data
            early_stop_callback = EarlyStopping(monitor='loss', min_delta=0.01, patience=5, verbose=1, mode='auto')
            callback_log_path = os.path.join(history_path, model.name + '-' + str(k) + '.log')
            with open(callback_log_path, 'w') as f:
                def print_fcn(s):
                    f.write(s)
                    f.write("\n")
            start = time.time()
            with open(callback_log_path, 'w') as f:
                def print_fcn(s):
                    f.write(s)
                    f.write("\n")

                model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                          callbacks=[early_stop_callback, LoggingCallback(print_fcn)])
            end = time.time()

            learn_time = end - start
            score = model.evaluate(X_test, Y_test, verbose=1)

            start = time.time()
            y_pred_prob = model.predict(X_test)
            end = time.time()
            predict_time = end - start
            y_pred = np.array([np.argmax(i) for i in y_pred_prob])

            y_test = np.array([np.argmax(i) for i in Y_test])
            cf = confusion_matrix(y_test, y_pred)
            p, r, f1, auc_roc = cm_measures(y_test, y_pred)

            print('Model: ', model.name, ', k=', k)
            print('P=', p, ', R=', r, 'F1=', f1)

            performance_results.append(
                [model.name, k, score[0], score[1], score[2], score[3], cf[1][1], cf[1][0], cf[0][1],
                 cf[0][0], p, r, f1, auc_roc, learn_time, predict_time])

            y_pred_prob = np.array([v[1] for v in y_pred_prob])

            precision_recall_curve(model.name, y_test, y_pred_prob, img_folder_path, k)

            # model.save(os.path.join(models_folder_path, model.name + 'k' + str(k)))
            # Reset learnt weights to execute new experiment
            model.set_weights(init_weights)
            k += 1
        del model
        gc.collect()
    import csv
    with open(results_path, 'w') as outcsv:
        # configure writer to write standard csv file
        writer = csv.writer(outcsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['Model', 'K', 'keras_loss', 'keras_acc', 'keras_mae', 'keras_bin_acc', 'tp', 'fn', 'fp', 'tn',
                         'P', 'R', 'F1', 'AUC-ROC', 'learning_time', 'test_time'])
        for item in performance_results:
            writer.writerow(item)


def experiments(X, Y, wv_model, test_size, batch_size, epochs, results_path, history_path, img_folder_path,
                models_folder_path, models_containers):
    performance_results = list()

    Y = np_utils.to_categorical(Y, 2)

    for model_container in models_containers:
        print(type(model_container).__name__)
        model, X_preprocessed = model_container.create_model(X, wv_model)
        X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y, test_size=test_size, random_state=7,
                                                            stratify=Y)
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', mae, binary_accuracy])
        print(model.summary())

        # Fit model on training data
        early_stop_callback = EarlyStopping(monitor='loss', min_delta=0.01, patience=5, verbose=1, mode='auto')
        callback_log_path = os.path.join(history_path, model.name + '.log')
        start = time.time()
        with open(callback_log_path, 'w') as f:
            def print_fcn(s):
                f.write(s)
                f.write("\n")
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                      callbacks=[early_stop_callback, LoggingCallback(print_fcn)])
        end = time.time()

        learn_time = end - start
        score = model.evaluate(X_test, Y_test, verbose=1)

        start = time.time()
        y_pred_prob = model.predict(X_test)
        end = time.time()
        predict_time = end - start
        y_pred = np.array([np.argmax(i) for i in y_pred_prob])

        y_test = np.array([np.argmax(i) for i in Y_test])
        cf = confusion_matrix(y_test, y_pred)
        p, r, f1, auc_roc = cm_measures(y_test, y_pred)

        print('Model: ', model.name)
        print('P=', p, ', R=', r, 'F1=', f1)

        performance_results.append(
            [model.name, score[0], score[1], score[2], score[3], cf[1][1], cf[1][0], cf[0][1],
             cf[0][0], p, r, f1, auc_roc, learn_time, predict_time])

        y_pred_prob = np.array([v[1] for v in y_pred_prob])

        precision_recall_curve(model.name, y_test, y_pred_prob, img_folder_path)

        # model.save(os.path.join(models_folder_path, model.name))

        del model
        gc.collect()

    import csv
    with open(results_path, 'w') as outcsv:
        # configure writer to write standard csv file
        writer = csv.writer(outcsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['Model', 'keras_loss', 'keras_acc', 'keras_mae', 'keras_bin_acc', 'tp', 'fn', 'fp', 'tn',
                         'P', 'R', 'F1', 'AUC-ROC', 'learning_time', 'test_time'])
        for item in performance_results:
            writer.writerow(item)


class LoggingCallback(Callback):

    def __init__(self, print_fcn=print,
                 format_epoch="Epoch: {} - {}",
                 format_keyvalue="{}: {:0.4f}",
                 format_separator=" - "):
        Callback.__init__(self)
        self.print_fcn = print_fcn
        self.format_epoch = format_epoch
        self.format_keyvalue = format_keyvalue
        self.format_separator = format_separator

    def on_epoch_end(self, epoch, logs={}):
        values = self.format_separator.join(self.format_keyvalue.format(k, v) for k, v in iteritems(logs))
        msg = self.format_epoch.format(epoch, values)
        self.print_fcn(msg)