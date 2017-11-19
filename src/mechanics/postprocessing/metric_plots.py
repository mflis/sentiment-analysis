import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_prec_recall(y_true, y_pred, destination):
    prec, recall, thresholds_pr = precision_recall_curve(y_true, y_pred)
    baseline = np.count_nonzero(y_true) / len(y_true)
    plt.figure()
    plt.axhline(y=baseline, ls="--", c=".3")
    plt.title('Precision/Recall curve')
    plt.ylabel("Precision")
    plt.xlabel('Recall')
    plt.plot(recall, prec)
    plt.savefig(destination, format='png')


def plot_roc_curve(y_true, y_pred, destination):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.figure()
    plt.title('ROC curve ')
    plt.xlabel("false positive rate")
    plt.ylabel('true positive rate')
    plt.plot(fpr, tpr)
    plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
    plt.savefig(destination, format='png')
