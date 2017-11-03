from sklearn.metrics import roc_auc_score
from tensorflow.contrib.keras.python.keras.callbacks import Callback


class AucMetric(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        score = roc_auc_score(self.y_val, y_pred)
        logs['auc'] = score
        print("\ninterval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
