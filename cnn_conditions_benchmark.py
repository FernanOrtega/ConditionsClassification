import os, sys
from data_loaders import load_dataset, load_wv_model
from experiments_launchers import experiments_with_K_fold
from model_creator import ModelContainer9, ModelContainer14, ModelContainer13, ModelContainer12, ModelContainer11, \
    ModelContainer10, ModelContainer7


def get_model_containers():
    models = list()
    models.append(ModelContainer7())
    models.append(ModelContainer9())
    models.append(ModelContainer10())
    models.append(ModelContainer11())
    models.append(ModelContainer12())
    models.append(ModelContainer13())
    models.append(ModelContainer14())

    return models


if __name__ == '__main__':

    if len(sys.argv) >= 2:
        import theano.sandbox.cuda
        theano.sandbox.cuda.use(sys.argv[1])

    dir_path = os.path.dirname(os.path.realpath(__file__))

    wv_model_path = os.path.join(dir_path, 'models/w2v-ciaoreviews')
    dataset_path = os.path.join(dir_path, 'datasets/dataset')
    labels_path = os.path.join(dir_path, 'datasets/dataset.labels')
    results_path = os.path.join(dir_path, 'results/benchmark-results-05-10-17.csv')
    history_path = os.path.join(dir_path, 'results/history/')
    img_folder_path = os.path.join(dir_path, 'results/img')
    models_folder_path = os.path.join(dir_path, 'models/')

    n_k_splits = 4
    batch_size = 30
    epochs = 50
    test_size = 0.25

    X, Y = load_dataset(dataset_path, labels_path)
    wv_model = load_wv_model(wv_model_path)

    # experiments(X, Y, wv_model, test_size, batch_size, epochs, results_path, history_path, img_folder_path,
    #                       models_folder_path, get_model_containers())
    experiments_with_K_fold(X, Y, wv_model, n_k_splits, batch_size, epochs, results_path, history_path, img_folder_path,
                            models_folder_path, get_model_containers())
