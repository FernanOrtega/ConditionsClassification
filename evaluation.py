import os

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Turning off interactive mode of PyPlot
plt.ioff()


def precision_recall_curve(model_name, y_test, y_pred, img_folder_path, k=-1):
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
    if k < 0:
        plt.title('2-class PR curve (model=' + str(model_name) + ': AUC={0:0.2f}'
                  .format(average_precision))
        plt.savefig(os.path.join(img_folder_path, 'p-r-curve-' + model_name + '.png'))
    else:
        plt.title('2-class PR curve (model=' + str(model_name) + ', k=' + str(k) + ': AUC={0:0.2f}'
                  .format(average_precision))
        plt.savefig(os.path.join(img_folder_path, 'p-r-curve-' + model_name + '-' + str(k) + '.png'))


def cm_measures(y_test, y_pred):
    return precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), \
           roc_auc_score(y_test, y_pred)