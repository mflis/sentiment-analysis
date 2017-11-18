from functools import partial

# noinspection PyUnresolvedReferences
import numpy as np
import tensorflow as tf
from sklearn.metrics import *


def _estimate_on_binarized_prediction(y_true, y_pred, metric):
    # todo consider custom threshold
    y_pred_int = np.rint(y_pred).astype(int)
    return metric(y_true, y_pred_int)


def recall(y_true, y_pred):
    metric_function = partial(_estimate_on_binarized_prediction, metric=recall_score)
    return tf.py_func(metric_function, [y_true, y_pred], tf.double)


def precision(y_true, y_pred):
    metric_function = partial(_estimate_on_binarized_prediction, metric=precision_score)
    return tf.py_func(metric_function, [y_true, y_pred], tf.double)


def roc_score(y_true, y_pred):
    metric_function = partial(_estimate_on_binarized_prediction, metric=roc_auc_score)
    return tf.py_func(metric_function, [y_true, y_pred], tf.double)
