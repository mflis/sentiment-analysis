import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
from tensorflow.contrib.keras.python.keras.callbacks import History

import src


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def show_confusion_matrix(y_true, y_pred):
    class_names = ['pos', 'neg']
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig("../confusions/{}.png".format(src.CURRENT_TIME), bbox_inches='tight')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig("../confusions/{}_normalized.png".format(src.CURRENT_TIME), bbox_inches='tight')


def plot_metric(history: History, name):
    y = history.history[name][1:]
    x = np.arange(2, len(y) + 2)
    plt.figure()
    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.plot(x, y, 'ro')
    plt.savefig("../metrics/{}_{}.png".format(src.CURRENT_TIME, name), bbox_inches='tight')


def plot_prec_recall(y_true, y_pred, name_suffix='tfidf-log'):
    prec, recall, thresholds_pr = precision_recall_curve(y_true, y_pred)
    plt.figure()
    plt.title('Precision/Recall curve - {}'.format(name_suffix))
    plt.ylabel("Precision")
    plt.xlabel('Recall')
    plt.plot(recall, prec)
    plt.savefig('../prec_recall_curves/{}.png'.format(src.CURRENT_TIME))


def plot_roc_curve(y_true, y_pred, name_suffix='tfidf-log'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.figure()
    plt.title('ROC curve - {}'.format(name_suffix))
    plt.xlabel("false positive rate")
    plt.ylabel('true positive rate')
    plt.plot(fpr, tpr)
    plt.savefig('../roc_curves/{}.png'.format(src.CURRENT_TIME))
