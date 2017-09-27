from typing import Tuple

import tensorflow as tf


# todo write some basic tests
def split(tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    train_frac = 0.8 * tensor.shape[0]
    val_test_frac = int(((1 - train_frac) / 2) * tensor.shape[0])
    return tf.split(tensor, [train_frac, val_test_frac, val_test_frac])


def get_batches(x: tf.Tensor, y: tf.Tensor, batch_size):
    # tf.train.batch([x, y], batch_size)
    n_batches = tf.shape(x)[0] // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]
