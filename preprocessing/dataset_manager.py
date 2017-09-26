import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim


def create_str_dataset(pathpos, pathneg, pathout_dataset, pathout_labels):
    dataset = list();
    labels = list();
    update_array(dataset, labels, pathpos, 1)
    update_array(dataset, labels, pathneg, 0)

    with open(pathout_dataset, 'w', encoding='utf-8') as f:
        for line in dataset:
            f.write(str(line))
    with open(pathout_labels, 'w', encoding='utf-8') as f:
        for line in labels:
            f.write(str(line)+'\n')


def update_array(dataset, labels, path, value_class):
    for line in open(path, encoding='utf-8'):
        # vline = [[get_we(we_model,t) for t in elem] for elem in eval(line)]
        # vline = [[t for t in elem] for elem in eval(line)]
        # dataset.append(vline)
        dataset.append(line)
        labels.append(value_class)


create_str_dataset('C:/datasets/es/positives.tok', 'C:/datasets/es/negatives.tok', 'C:/datasets/es/dataset',
                   'C:/datasets/es/dataset.labels')