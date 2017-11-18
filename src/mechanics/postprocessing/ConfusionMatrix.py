import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


# import os

# import src
# from src.mechanics.postprocessing import RESULTS_DIR


class ConfusionMatrix:
    # noinspection PyDefaultArgument
    def __init__(self, y_true, y_pred, class_names=['high\nscore', 'low\nscore']):
        self.y_true = y_true
        self.y_pred = np.rint(y_pred).astype(int)
        self.class_names = class_names

    def prepare_confusion_matrix(self, cm, classes,
                                 normalize=False,
                                 title='Confusion matrix',
                                 cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # plt.rcParams.update({'font.size': 22})
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        self.write_counts_on_matrix(cm, normalize)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def write_matrix(self, destination):
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.y_true, self.y_pred)

        # todo is this really needed?
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.prepare_confusion_matrix(cnf_matrix, classes=self.class_names, title="")
        plt.savefig(destination, bbox_inches='tight')

    @staticmethod
    def write_counts_on_matrix(cm, normalize):
        """
        prints raw counts on normalized values on confusion matrix
        :param cm: color used to plot matrix
        :param normalize: are values raw counts or normalized
        :return:
        """
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

            # def show_confusion_matrix(self):
            #
            #     # Plot non-normalized confusion matrix
            #     plt.figure()
            #     self.prepare_confusion_matrix(cnf_matrix, classes=self.class_names, title="")
            #     # todo fix paths
            #     target_dir = os.path.join(RESULTS_DIR, 'confusions')
            #     plt.savefig("../confusions/{}.png".format(src.CURRENT_TIME), bbox_inches='tight')
            #
            #     # Plot normalized confusion matrix
            #     plt.figure()
            #     self.prepare_confusion_matrix(cnf_matrix, classes=self.class_names, normalize=True)
            #     plt.savefig("../confusions/{}_normalized.png".format(src.CURRENT_TIME), bbox_inches='tight')
