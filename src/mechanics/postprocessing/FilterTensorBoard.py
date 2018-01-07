import re

from tensorflow.python import keras


class FilterTensorBoard(keras.callbacks.TensorBoard):
    """
    Write out only certain logs to a specific directory
    Intended to separate train/validation logs
    Keras adds "val_" to the beginning of all the validation metrics
    so we can include (or exclude) those

    source: https://github.com/jsilter/dbpedia_classify/blob/part3/custom_callbacks.py
    """

    def __init__(self, *args, **kwargs):
        self.log_regex = kwargs.pop('log_regex', '.*')
        # Dictionary for string replacement
        self.rep_dict = kwargs.pop('rep_dict', {'val_': ''})
        super(FilterTensorBoard, self).__init__(*args, **kwargs)

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
