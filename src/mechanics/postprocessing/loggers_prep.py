from sacred import Ingredient

from mechanics.postprocessing.ConfusionMatrix import ConfusionMatrix
from mechanics.postprocessing.ImagesTensorBoard import ImagesTensorBoard
from mechanics.postprocessing.metric_plots import plot_roc_curve, plot_prec_recall

loggers_ingredient = Ingredient('loggers')


@loggers_ingredient.config
def conf():
    from definitions import ROOT_DIR, CURRENT_TIME
    import os
    log_dir = 'results/logs1'
    train_log_dir = os.path.join(ROOT_DIR, '{}/{}-train'.format(log_dir, CURRENT_TIME))
    val_log_dir = os.path.join(ROOT_DIR, '{}/{}-validation'.format(log_dir, CURRENT_TIME))


@loggers_ingredient.capture
def prepare_loggers(train_log_dir, val_log_dir):
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
