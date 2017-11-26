import os

from sacred import Ingredient
from tensorflow.python.keras._impl.keras.callbacks import ModelCheckpoint

from definitions import ROOT_DIR, CURRENT_TIME
from mechanics.postprocessing.ConfusionMatrix import ConfusionMatrix
from mechanics.postprocessing.ImagesTensorBoard import ImagesTensorBoard
from mechanics.postprocessing.metric_plots import plot_roc_curve, plot_prec_recall

loggers_ingredient = Ingredient('loggers')


@loggers_ingredient.config
def conf():
    log_dir = 'results/logs1'
    checkpoints_dir = 'results/checkpoints'


@loggers_ingredient.capture
def loggers(log_dir):
    train_log_dir = os.path.join(ROOT_DIR, '{}/{}-train'.format(log_dir, CURRENT_TIME))
    val_log_dir = os.path.join(ROOT_DIR, '{}/{}-validation'.format(log_dir, CURRENT_TIME))
    plots = {'roc-train': plot_roc_curve,
             'prec-recall-train': plot_prec_recall,
             'confusion-metrix-train': lambda y_true, y_pred, dest: ConfusionMatrix(y_true, y_pred).write_matrix(dest)}
    train_tboard_logger = ImagesTensorBoard(plotting_functions=plots,
                                            suffix='train',
                                            log_dir=train_log_dir,
                                            write_graph=True,
                                            write_images=False, log_regex=r'^(?!val).*')
    val_tboard_logger = ImagesTensorBoard(plotting_functions=plots,
                                          suffix='validation',
                                          log_dir=val_log_dir, write_graph=False,
                                          write_images=False, log_regex=r"^val")
    return [train_tboard_logger, val_tboard_logger]


@loggers_ingredient.capture
def checkpoints(checkpoints_dir):
    file_partial = "{}-model".format(CURRENT_TIME)
    checkpoint_file = file_partial + "model.{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint_base = os.path.join(ROOT_DIR, checkpoints_dir)
    path = os.path.join(checkpoint_base, checkpoint_file)
    # save_weights_only=True - workaround for problem  with saving model with lambda layer
    # http://forums.fast.ai/t/unable-to-save-model-checkpoints-when-using-lambda-in-model/827/7
    return [ModelCheckpoint(path, save_weights_only=True, save_best_only=True)]
