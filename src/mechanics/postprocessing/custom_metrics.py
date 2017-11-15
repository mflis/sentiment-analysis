# noinspection PyUnresolvedReferences
import re
from functools import partial

# noinspection PyUnresolvedReferences
import numpy as np
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.metrics import recall_score, precision_recall_fscore_support
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import Callback

from src.mechanics.postprocessing.plots import *


def precision(y_true, y_pred):
    return tf.metrics.precision(y_true, y_pred)


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


def estimate_on_binarized_prediction(y_true, y_pred, metric):
    y_pred_int = np.rint(y_pred).astype(int)
    return metric(y_true, y_pred_int)


def recall(y_true, y_pred):
    aser = partial(estimate_on_binarized_prediction, metric=recall_score)
    return tf.py_func(aser, [y_true, y_pred], tf.double)


import io
import matplotlib.pyplot as plt


class FilterTensorBoard(keras.callbacks.TensorBoard):
    """
    Write out only certain logs to a specific directory
    Intended to separate train/validation logs
    Keras adds "val_" to the beginning of all the validation metrics
    so we can include (or exclude) those
    """

    def __init__(self, *args, **kwargs):
        self.log_regex = kwargs.pop('log_regex', '.*')
        # Dictionary for string replacement
        self.rep_dict = kwargs.pop('rep_dict', {'val_': ''})
        super(FilterTensorBoard, self).__init__(*args, **kwargs)

    def gen_plot(self):
        """Create a pyplot plot and save to buffer."""
        plt.figure()
        plt.plot([1, 2])
        plt.title("test")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    def filter_logs(self, logs):
        logs = logs or {}
        out_logs = {}

        for key in logs:
            if self.log_regex is None or re.match(self.log_regex, key):
                out_key = key
                for rep_key, rep_val in self.rep_dict.items():
                    out_key = out_key.replace(rep_key, rep_val, 1)
                out_logs[out_key] = logs[key]
        return out_logs

    def on_epoch_end(self, epoch, logs=None):
        super(FilterTensorBoard, self).on_epoch_end(epoch, self.filter_logs(logs))
        # Prepare the plot
        plot_buf = self.gen_plot()

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        # Add image summary
        summary_op = tf.summary.image("plot", image)

        # Session
        with tf.Session() as sess:
            # Run
            summary = sess.run(summary_op)
            # Write summary
            self.writer.add_summary(summary)
            self.writer.flush()
