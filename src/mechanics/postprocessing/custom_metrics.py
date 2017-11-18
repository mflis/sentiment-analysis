# noinspection PyUnresolvedReferences
import re

# noinspection PyUnresolvedReferences
import numpy as np
import tensorflow as tf
from sklearn.metrics import *
from tensorflow.python.keras.callbacks import Callback

from src.mechanics.postprocessing.ConfusionMatrix import *


class CustomMetrics(Callback):
    def __init__(self, tags):
        super(Callback, self).__init__()
        self.tags = tags

    def evaluate(self):
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_pred_float = self.model.predict(x_val, verbose=0)
        y_pred = np.rint(y_pred_float).astype(int)
        return y_val, y_pred, y_pred_float

    def on_train_end(self, logs=None):
        y_true, y_pred_class, y_pred_prob = self.evaluate()
        show_confusion_matrix(y_true, y_pred_class)

        prec, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_prob)
        plt.figure()
        plt.title('Prcision/Recall curve')
        plt.ylabel("Precision")
        plt.xlabel('Recall')
        plt.plot(recall, prec)
        plt.savefig('../prec_recall_curves/{}-{}.png'.format(src.CURRENT_TIME, self.tags))

    def on_epoch_end(self, epoch, logs=None):
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_pred_float = self.model.predict(x_val, verbose=0)
        y_pred = np.rint(y_pred_float).astype(int)
        auc = roc_auc_score(y_val, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val, y_pred, average='binary')
        kappa = cohen_kappa_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)

        logs['auc'] = auc
        logs['precision'] = precision
        logs['recall'] = recall
        logs['f1_score'] = f1
        logs['support'] = support
        logs['kappa'] = kappa
        logs['confusion_matrix'] = conf_matrix
        tf.summary.scalar("precision", precision)
        print("\nAUC: {:.6f}, ".format(auc))

